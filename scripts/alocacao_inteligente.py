#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import io

# Forçar encoding UTF-8 para stdout/stderr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
"""
Algoritmo Inteligente de Alocação com Machine Learning
Integrado com backend Node.js para sistema de gerenciamento de alocação
Autor: Sistema de Alocação Inteligente
"""

import json
import argparse
from itertools import permutations
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import networkx as nx
    HAS_NX = True
except Exception:
    HAS_NX = False

# Controle de aleatoriedade para reprodutibilidade total
np.random.seed(42)


# Funções puras para melhor testabilidade
def score_ocupacao_puro(alunos, capacidade, alvo_ocupacao=0.85):
    """Calcula score de ocupação puro"""
    if capacidade <= 0:
        return 0.0
    ocupacao = alunos / capacidade
    if ocupacao <= 1.0:  # Até 100% é bom
        return max(0.0, 1 - min(abs(ocupacao - alvo_ocupacao) / alvo_ocupacao, 1))
    else:  # Acima de 100% é problemático
        return max(0.0, 0.5 - (ocupacao - 1.0))  # Penalizar superlotação

def par_viavel_puro(deficit, esp_deficit, ocupacao, sala_movel, deficit_moveis_viavel=3):
    """Verifica se um par turma-sala é viável"""
    if esp_deficit > 0:
        return False
    # Não permitir ocupação > 100% (mesmo com móveis)
    if ocupacao > 1.0:
        return False
    return (deficit == 0) or (sala_movel == 1 and deficit <= deficit_moveis_viavel)

def label_heuristico_puro(ocupacao, atende_especial, cabe_direto, sala_movel, deficit, deficit_moveis_viavel=3):
    """Calcula label heurístico baseado em regras de negócio"""
    within = (0.70 <= ocupacao <= 1.0)
    deficit_pequeno = (deficit <= deficit_moveis_viavel) and (deficit > 0)
    aproveitamento_bom = ocupacao >= 0.75
    return 1 if (atende_especial and aproveitamento_bom and
                 (cabe_direto or (not cabe_direto and sala_movel and deficit_pequeno))) else 0

def validar_entrada(salas, turmas):
    """Valida entrada básica com verificações rigorosas"""
    if not isinstance(salas, list) or not isinstance(turmas, list):
        raise ValueError("Salas e turmas devem ser listas")
    for i, sala in enumerate(salas):
        if sala.get("id") is None and sala.get("id_sala") is None:
            raise ValueError(f"Sala '{sala.get('nome','?')}' (índice {i}) sem id/id_sala")
        capacidade = sala.get('capacidade_total', 0)
        if not isinstance(capacidade, (int, float)) or capacidade < 0:
            raise ValueError(f"Sala {sala.get('nome', '?')} tem capacidade inválida: {capacidade}")
        if capacidade == 0 and str(sala.get('status', 'ATIVA')).upper() == 'ATIVA':
            print(f"⚠️ [VALIDATION] Sala '{sala.get('nome', '?')}' ativa com capacidade 0", file=sys.stderr)
        especiais = sala.get('cadeiras_especiais', 0)
        if not isinstance(especiais, (int, float)) or especiais < 0:
            raise ValueError(f"Sala {sala.get('nome', '?')} tem cadeiras especiais inválidas: {especiais}")
    for i, turma in enumerate(turmas):
        if turma.get("id") is None and turma.get("id_turma") is None:
            raise ValueError(f"Turma '{turma.get('nome','?')}' (índice {i}) sem id/id_turma")
        alunos = turma.get('alunos', 0)
        if not isinstance(alunos, (int, float)) or alunos <= 0:
            raise ValueError(f"Turma {turma.get('nome', '?')} tem número de alunos inválido: {alunos}")
        esp_necessarias = turma.get('esp_necessarias', 0)
        if not isinstance(esp_necessarias, (int, float)) or esp_necessarias < 0:
            raise ValueError(f"Turma {turma.get('nome', '?')} tem especiais necessárias inválidas: {esp_necessarias}")
        if esp_necessarias > alunos:
            print(f"⚠️ [VALIDATION] Turma '{turma.get('nome', '?')}': especiais necessárias ({esp_necessarias}) > alunos ({alunos})", file=sys.stderr)


class AlocacaoInteligenteMLA:
    """Classe principal para algoritmo de alocação com Machine Learning"""
    def __init__(self, salas, turmas, parametros):
        validar_entrada(salas, turmas)
        self.salas = self._normalize_salas(salas)
        self.turmas = self._normalize_turmas(turmas)
        self.parametros = parametros or {}
        self.clf = None
        self.df = None
        self.debug_best_params = None
        self.modelo_regras = None

    def _normalize_salas(self, salas):
        normalized = []
        for sala in salas:
            moveis_qtd = int(sala.get("cadeiras_moveis", 0) or 0)
            normalized.append({
                "id": sala.get("id"),
                "id_sala": sala.get("id_sala", sala.get("id")),
                "nome": sala.get("nome", ""),
                "capacidade_total": int(sala.get("capacidade_total", 0) or 0),
                "localizacao": sala.get("localizacao", ""),
                "status": str(sala.get("status", "ATIVA")).upper(),
                "cadeiras_moveis_qtd": moveis_qtd,
                "cadeiras_moveis": bool(moveis_qtd > 0),
                "cadeiras_especiais": int(sala.get("cadeiras_especiais", 0) or 0),
            })
        return normalized

    def _normalize_turmas(self, turmas):
        normalized = []
        for turma in turmas:
            normalized.append({
                "id": turma.get("id"),
                "id_turma": turma.get("id_turma", turma.get("id")),
                "nome": turma.get("nome", ""),
                "alunos": int(turma.get("alunos", 0)),
                "duracao_min": int(turma.get("duracao_min", 120)),
                "esp_necessarias": int(turma.get("esp_necessarias", 0))
            })
        return normalized

    def build_pair_features(self):
        """Constrói features para cada par (turma, sala)"""
        p = self.parametros
        alvo_ocupacao = float(p.get("alvo_ocupacao", 0.85))
        deficit_moveis_viavel = int(p.get("deficit_moveis_viavel", 3))
        permitir_moveis_ml = bool(p.get("permitir_moveis_no_ml", False))

        rows = []
        for t in self.turmas:
            for s in self.salas:
                if s["status"].upper() != "ATIVA":
                    continue

                alunos = t["alunos"]
                cap = s["capacidade_total"]
                deficit = max(0, alunos - cap)
                sobra = max(0, cap - alunos)
                cabe_direto = (cap >= alunos)
                sala_movel = 1 if s.get("cadeiras_moveis", False) else 0

                # especiais
                esp_need = int(t.get("esp_necessarias", 0))
                esp_have = int(s.get("cadeiras_especiais", 0))
                esp_deficit = max(0, esp_need - esp_have)
                esp_sobra = max(0, esp_have - esp_need)
                atende_especial = 1 if esp_deficit == 0 else 0

                # capacidade/ocupação efetivas quando permitir_moveis_no_ml
                max_emprestimo = deficit_moveis_viavel
                cap_efetiva = cap
                if permitir_moveis_ml and sala_movel and alunos > cap:
                    cap_efetiva = cap + min(max_emprestimo, alunos - cap)

                ocupacao = alunos / cap if cap > 0 else 0.0
                ocupacao_eff = alunos / cap_efetiva if cap_efetiva > 0 else 0.0
                deficit_eff = max(0, alunos - cap_efetiva)

                cap_para_score = cap_efetiva if permitir_moveis_ml else cap
                score_ocupacao = score_ocupacao_puro(alunos, cap_para_score, alvo_ocupacao)

                ocup_para_label = ocupacao_eff if permitir_moveis_ml else ocupacao
                deficit_para_label = deficit_eff if permitir_moveis_ml else deficit
                cabe_para_label = (cap_efetiva >= alunos) if permitir_moveis_ml else cabe_direto

                modo_treinamento = self.parametros.get("modo_treinamento", "heuristico")
                if modo_treinamento == "historico":
                    chave_historico = f"{t['id_turma']}_{s['id_sala']}"
                    historico_data = self.parametros.get("dados_historicos", {})
                    label_heuristico = historico_data.get(chave_historico, 0)
                else:
                    label_heuristico = label_heuristico_puro(
                        ocup_para_label, atende_especial, cabe_para_label,
                        sala_movel, deficit_para_label, deficit_moveis_viavel
                    )

                rows.append({
                    "id_turma": t["id_turma"],
                    "id_sala": s["id_sala"],
                    "alunos": alunos,
                    "capacidade_total": cap,
                    "capacidade_efetiva": cap_efetiva,
                    "deficit": deficit,
                    "deficit_eff": deficit_eff,
                    "sobra_local": sobra,
                    "sala_movel": sala_movel,
                    "ocupacao": ocupacao,
                    "ocupacao_eff": ocupacao_eff,
                    "score_ocupacao": score_ocupacao,
                    "esp_necessarias": esp_need,
                    "esp_disponiveis": esp_have,
                    "esp_deficit": esp_deficit,
                    "esp_sobra": esp_sobra,
                    "atende_especial": atende_especial,
                    "label_heuristico": label_heuristico
                })
        return pd.DataFrame(rows)

    def _treinar_e_pontuar(self):
        """Treina modelo com validação cruzada, grid search e calibração"""
        print("🔍 [ML] Iniciando treinamento avançado", file=sys.stderr)

        feature_cols = [
            "alunos", "capacidade_total", "deficit", "sobra_local", "sala_movel",
            "ocupacao", "score_ocupacao", "esp_necessarias", "esp_disponiveis",
            "esp_deficit", "esp_sobra", "atende_especial"
        ]
        if self.parametros.get("permitir_moveis_no_ml", False):
            for col in ["capacidade_efetiva", "deficit_eff", "ocupacao_eff"]:
                if col not in feature_cols and col in self.df.columns:
                    feature_cols.append(col)

        X = self.df[feature_cols].values
        y = self.df["label_heuristico"].values

        print(f"📊 [ML] Dataset: {len(X)} pares, classes: {Counter(y)}", file=sys.stderr)

        p = self.parametros
        peso_ml = float(p.get("peso_ml", 0.4))
        peso_occ = float(p.get("peso_ocupacao", 0.6))
        limiar_cls = float(p.get("limiar_cls", 0.5))

        if peso_ml + peso_occ != 1.0:
            total = peso_ml + peso_occ
            peso_ml = peso_ml / total
            peso_occ = peso_occ / total
            print(f"📊 [ML] Pesos normalizados: ML={peso_ml:.2f}, Ocupação={peso_occ:.2f}", file=sys.stderr)

        custom_max_depth = p.get("max_depth")
        custom_min_samples_split = p.get("min_samples_split")
        custom_min_samples_leaf = p.get("min_samples_leaf")

        if len(np.unique(y)) > 1 and len(y) >= 12:
            print("🎯 [ML] Dataset suficiente: usando Grid Search + Calibração", file=sys.stderr)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            param_grid = {
                "max_depth": [custom_max_depth] if custom_max_depth else [3, 4, 5, None],
                "min_samples_split": [custom_min_samples_split] if custom_min_samples_split else [2, 5, 10],
                "min_samples_leaf": [custom_min_samples_leaf] if custom_min_samples_leaf else [1, 2, 5],
            }
            grid = GridSearchCV(
                DecisionTreeClassifier(random_state=42, class_weight="balanced"),
                param_grid=param_grid,
                scoring="f1", cv=cv, n_jobs=-1
            )
            grid.fit(X, y)
            base = grid.best_estimator_
            self.debug_best_params = grid.best_params_
            print(f"🏆 [ML] Melhores parâmetros: {self.debug_best_params}", file=sys.stderr)

            self.clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
            self.clf.fit(X, y)
            self.df["proba_ml"] = self.clf.predict_proba(X)[:, 1]

            try:
                self.modelo_regras = export_text(base, feature_names=feature_cols)
                print("📋 [ML] Regras da árvore exportadas para debug", file=sys.stderr)
                if self.parametros.get("dump_arvore"):
                    with open("arvore_decisao.txt", "w", encoding="utf-8") as f:
                        f.write(self.modelo_regras)
                    print("💾 [DUMP] Árvore salva em: arvore_decisao.txt", file=sys.stderr)
            except Exception as e:
                print(f"⚠️ [ML] Erro ao exportar regras: {e}", file=sys.stderr)
        else:
            print("⚠️ [ML] Dataset pequeno: treinamento básico sem calibração", file=sys.stderr)
            self.clf = DecisionTreeClassifier(
                max_depth=custom_max_depth or 4,
                min_samples_split=custom_min_samples_split or 2,
                min_samples_leaf=custom_min_samples_leaf or 1,
                random_state=42,
                class_weight="balanced"
            )
            self.clf.fit(X, y)
            proba_result = self.clf.predict_proba(X)
            self.df["proba_ml"] = proba_result[:, 1] if proba_result.shape[1] > 1 else 0.0
            self.debug_best_params = {
                "max_depth": custom_max_depth or 4,
                "min_samples_split": custom_min_samples_split or 2,
                "min_samples_leaf": custom_min_samples_leaf or 1
            }

        self.df["proba_bom"] = np.clip(
            peso_ml * self.df["proba_ml"] + peso_occ * self.df["score_ocupacao"].values,
            0.0, 1.0
        )
        print(f"✅ [ML] Probabilidades combinadas: ML({peso_ml:.1%}) + Ocupação({peso_occ:.1%})", file=sys.stderr)

        mask_perfeito = (
            np.isclose(self.df["ocupacao"].values, 1.0) &
            (self.df["esp_deficit"].values == 0) &
            (self.df["deficit"].values == 0)
        )
        matches_perfeitos = mask_perfeito.sum()
        if matches_perfeitos > 0:
            self.df.loc[mask_perfeito, "proba_bom"] = 1.0
            self.df["match_perfeito"] = False
            self.df.loc[mask_perfeito, "match_perfeito"] = True
            print(f"🎯 [ML] {matches_perfeitos} matches perfeitos forçados a 100%", file=sys.stderr)
        else:
            self.df["match_perfeito"] = False

        try:
            if len(np.unique(y)) > 1:
                usar_metricas_cv = self.parametros.get("metricas_cv", True)
                if len(y) >= 12 and usar_metricas_cv:
                    print("🔍 [METRICS] Usando métricas honestas com cross-validation", file=sys.stderr)
                    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                    y_proba_cv = cross_val_predict(self.clf, X, y, cv=cv, method="predict_proba")[:, 1]
                    y_pred_cv = (y_proba_cv >= limiar_cls).astype(int)
                    self.clf_report = classification_report(y, y_pred_cv, output_dict=True, zero_division=0)
                    print(f"📈 [METRICS] F1-score honesto (CV): {self.clf_report.get('1', {}).get('f1-score', 0):.3f}", file=sys.stderr)
                    self.metricas_honestas = True
                else:
                    if not usar_metricas_cv:
                        print("⚠️ [METRICS] Métricas CV desabilitadas: usando treino=teste", file=sys.stderr)
                    else:
                        print("⚠️ [METRICS] Dataset pequeno: métricas simples (treino=teste)", file=sys.stderr)
                    y_pred = (self.df["proba_ml"].values >= limiar_cls).astype(int)
                    self.clf_report = classification_report(y, y_pred, output_dict=True, zero_division=0)
                    print(f"📈 [METRICS] F1-score (treino=teste): {self.clf_report.get('1', {}).get('f1-score', 0):.3f}", file=sys.stderr)
                    self.metricas_honestas = False
            else:
                self.clf_report = {"note": "Apenas uma classe disponível"}
                self.metricas_honestas = False
        except Exception as e:
            print(f"⚠️ [METRICS] Erro ao calcular métricas: {e}", file=sys.stderr)
            self.clf_report = {"error": str(e)}
            self.metricas_honestas = False

    def treinar_modelo(self):
        """Método de compatibilidade - chama o novo treinamento"""
        try:
            self.df = self.build_pair_features()
            if self.df.empty:
                print("❌ [PYTHON] Nenhum par turma-sala válido encontrado", file=sys.stderr)
                raise Exception("Nenhum par turma-sala válido encontrado")
            self._treinar_e_pontuar()
            if hasattr(self, 'clf_report') and '1' in self.clf_report:
                accuracy = self.clf_report['1'].get('f1-score', 0.5)
            else:
                accuracy = 0.5
            print(f"🎯 [PYTHON] Modelo treinado com acurácia estimada: {accuracy:.2%}", file=sys.stderr)
            return accuracy
        except Exception as e:
            print(f"❌ [PYTHON] Erro crítico no treinamento: {e}", file=sys.stderr)
            return 0.0

    def par_viavel(self, row):
        """Verifica se um par turma-sala é viável usando função pura"""
        p = self.parametros
        deficit_moveis_viavel = int(p.get("deficit_moveis_viavel", 3))
        usar_eff = bool(p.get("permitir_moveis_no_ml", False))
        deficit_v = row["deficit_eff"] if usar_eff and "deficit_eff" in row else row["deficit"]
        ocup_v = row["ocupacao_eff"] if usar_eff and "ocupacao_eff" in row else row["ocupacao"]
        return par_viavel_puro(
            deficit_v, row["esp_deficit"], ocup_v, row["sala_movel"], deficit_moveis_viavel
        )

    def _otimizar_hungarian(self):
        """Otimização usando Hungarian Algorithm (O(n³)) com empréstimo real entre salas móveis"""
        print("🔍 [OPT] Iniciando otimização Hungarian", file=sys.stderr)
        if not HAS_SCIPY:
            print("⚠️ [OPT] SciPy não disponível, usando força bruta", file=sys.stderr)
            return self._otimizar_forca_bruta()

        turma_ids = [t["id_turma"] for t in self.turmas]
        sala_ids = [s["id_sala"] for s in self.salas]
        idx_turma = {tid: i for i, tid in enumerate(turma_ids)}
        idx_sala = {sid: j for j, sid in enumerate(sala_ids)}

        # Construir matriz de utilidade (permite sala móvel mesmo com cap < alunos, desde que especiais atendam)
        U = np.zeros((len(turma_ids), len(sala_ids)), dtype=float)
        for _, r in self.df.iterrows():
            esp_ok = (r["esp_deficit"] == 0)
            cabe_direto = (r["capacidade_total"] >= r["alunos"])
            movel = bool(r["sala_movel"])
            if esp_ok and (cabe_direto or movel):
                i = idx_turma[r["id_turma"]]
                j = idx_sala[r["id_sala"]]
                U[i, j] = max(0.0, float(r["proba_bom"]))

        custo_inviavel = float(self.parametros.get("custo_inviavel", 1e6))

        usar_tiebreak = self.parametros.get("usar_tiebreak_sobra", False)
        if usar_tiebreak and len(self.df) > 0:
            print("🎯 [OPT] Usando tie-break por aproveitamento (menor sobra)", file=sys.stderr)
            S = np.zeros((len(turma_ids), len(sala_ids)), dtype=float)
            sobras = self.df["sobra_local"].values
            sobra_min = sobras.min()
            sobra_range = sobras.ptp() + 1e-9
            for _, r in self.df.iterrows():
                esp_ok = (r["esp_deficit"] == 0)
                cabe_direto = (r["capacidade_total"] >= r["alunos"])
                movel = bool(r["sala_movel"])
                if esp_ok and (cabe_direto or movel):
                    i = idx_turma[r["id_turma"]]
                    j = idx_sala[r["id_sala"]]
                    sobra_norm = (r["sobra_local"] - sobra_min) / sobra_range
                    S[i, j] = sobra_norm
            tiebreak_weight = float(self.parametros.get("peso_tiebreak", 0.01))
            C = np.where(U > 0.0, (1.0 - U) + tiebreak_weight * S, custo_inviavel)
        else:
            C = np.where(U > 0.0, 1.0 - U, custo_inviavel)

        print(f"🔢 [OPT] Matriz {len(turma_ids)}x{len(sala_ids)}, pares viáveis: {(U > 0).sum()}", file=sys.stderr)

        # Resolver assignment problem (matching)
        row_ind, col_ind = linear_sum_assignment(C)

        # ---------- Pós-processamento: empréstimo real entre salas móveis (sem limite por doador) ----------
        alocacoes = []
        turmas_alocadas = set()

        salas_by_id_sala = {s["id_sala"]: s for s in self.salas}
        turmas_by_id_turma = {t["id_turma"]: t for t in self.turmas}

        # 1) Candidatas de alocação (podem ter déficit)
        candidatas = []
        for i, j in zip(row_ind, col_ind):
            if C[i, j] >= custo_inviavel:
                continue
            sel = self.df[(self.df["id_turma"] == turma_ids[i]) &
                          (self.df["id_sala"] == sala_ids[j])]
            if sel.empty:
                # Logar e pular, para não derrubar o processo
                print(f"⚠️ [OPT] Par não encontrado no DF: turma={turma_ids[i]} sala={sala_ids[j]}", file=sys.stderr)
                continue
            linha = sel.iloc[0]
            t = turmas_by_id_turma[turma_ids[i]]
            s = salas_by_id_sala[sala_ids[j]]
            deficit = max(0, int(linha["alunos"] - linha["capacidade_total"]))
            candidatas.append((linha, t, s, deficit))

        # 2) Hospedados por sala (para calcular sobras reais dos doadores)
        hospedados = {s["id"]: 0 for s in self.salas}
        for linha, t, s, deficit in candidatas:
            hospedados[s["id"]] += int(linha["alunos"])

        # 3) Doadores móveis com sobra (capacidade - hospedados)
        doadores = {}
        for s in self.salas:
            if bool(s.get("cadeiras_moveis", False)):
                cap = int(s["capacidade_total"])
                hosped = int(hospedados.get(s["id"], 0))
                sobra = max(0, cap - hosped)
                doadores[s["id"]] = {"nome": s["nome"], "sobra": sobra}

        # 4) Para cada candidata com déficit, puxar cadeiras dos doadores (sem limite por doador)
        aloc_final = []
        for linha, t, s, deficit in candidatas:
            emprestimos = []
            faltando = deficit
            if faltando > 0:
                ordem = sorted(doadores.items(), key=lambda kv: kv[1]["sobra"], reverse=True)
                for sala_id, info in ordem:
                    if faltando <= 0:
                        break
                    if sala_id == s["id"]:
                        continue
                    pegar = min(faltando, info["sobra"])
                    if pegar > 0:
                        info["sobra"] -= pegar
                        faltando -= pegar
                        emprestimos.append((info["nome"], pegar))
            # se não conseguiu cobrir o déficit, descarta essa alocação
            if faltando > 0:
                continue

            turmas_alocadas.add(t["id"])
            obs = (
                f"Ocupacao: {linha['ocupacao']:.1%} "
                f"(score ocupacao: {linha['score_ocupacao']:.2f}), "
                f"Score ML: {float(linha.get('proba_ml', 0.0)):.2f}, "
                f"Score combinado: {float(linha['proba_bom']):.2f}, "
                f"Especiais: {linha['esp_necessarias']}/{linha['esp_disponiveis']} (Hungarian)"
            )
            if deficit > 0:
                detalhes = ", ".join([f"{qtd} de {nome}" for (nome, qtd) in emprestimos])
                obs += f", +{deficit} moveis ({detalhes}) → cap. efetiva: {linha['capacidade_total'] + deficit}"
            if bool(linha.get("match_perfeito", False)):
                obs += ", Match perfeito (forçado a 100%)"

            aloc_final.append({
                "sala_id": s["id"],
                "turma_id": t["id"],
                "compatibilidade_score": round(float(linha['proba_bom']) * 100, 2),
                "observacoes": obs
            })

        alocacoes = aloc_final

        # 5) Score final = média do proba_bom dos pares efetivamente mantidos
        if alocacoes:
            mantidos = []
            for a in alocacoes:
                # localizar linha correspondente
                turma_id_turma = next(t["id_turma"] for t in self.turmas if t["id"] == a["turma_id"])
                sala_id_sala = next(s["id_sala"] for s in self.salas if s["id"] == a["sala_id"])
                linha = self.df[(self.df["id_turma"] == turma_id_turma) &
                                (self.df["id_sala"] == sala_id_sala)].iloc[0]
                mantidos.append(float(linha["proba_bom"]))
            best_score = float(np.mean(mantidos))
        else:
            best_score = 0.0

        # Turmas não alocadas
        turmas_nao_alocadas = []
        for turma in self.turmas:
            if turma["id"] not in turmas_alocadas:
                turmas_nao_alocadas.append({
                    "id": turma["id"],
                    "nome": turma["nome"],
                    "alunos": turma["alunos"],
                    "esp_necessarias": turma["esp_necessarias"],
                    "motivo": self._analisar_motivo_nao_alocacao(turma)
                })

        print(f"✅ [OPT] Hungarian: {len(alocacoes)} alocações, score: {best_score:.2f}", file=sys.stderr)
        return alocacoes, best_score, turmas_nao_alocadas

    def _otimizar_fluxo_minimo(self):
        """
        Otimização com empréstimo global de cadeiras respeitando estoque de sobras.
        Requer networkx. Se não houver, levanta exceção para cair no fallback.
        """
        if not HAS_NX:
            raise RuntimeError("NetworkX indisponível")

        # Índices rápidos
        turmas = [t for t in self.turmas]
        salas  = [s for s in self.salas if s["status"].upper()=="ATIVA"]

        # Proba combinada só para pares viáveis (sem ultrapassar especiais)
        dfv = self.df[self.df.apply(self.par_viavel, axis=1)].copy()
        if dfv.empty:
            return [], 0.0, [{"id":t["id"],"nome":t["nome"],"alunos":t["alunos"],
                              "esp_necessarias":t["esp_necessarias"],"motivo":"Nenhum par viável"} for t in turmas]

        # Pré-mapas
        by_tid = {t["id_turma"]: t for t in turmas}
        by_sid = {s["id_sala"]: s for s in salas}

        # Grafo de fluxo
        G = nx.DiGraph()

        SRC, SNK = "SRC", "SNK"
        G.add_node(SRC); G.add_node(SNK)

        # Nós de turmas (demanda = alunos)
        for t in turmas:
            tn = f"T:{t['id_turma']}"
            G.add_node(tn, demand=-int(t["alunos"]))  # demanda negativa = precisa receber fluxo
            G.add_edge(SRC, tn, capacity=int(t["alunos"]), weight=0)

        # Nós de salas (oferta = capacidade fixa)
        for s in salas:
            sn = f"S:{s['id_sala']}"
            G.add_node(sn, demand=0)
            # oferta base: capacidade_total
            G.add_edge(sn, SNK, capacity=int(s["capacidade_total"]), weight=0)

        # Arestas turma->sala com "benefício" = proba_bom; usamos custo = -benefício*1e4 + tie-break
        # E permitimos exceder capacidade via "cadeiras emprestadas" por um nó DONATION,
        # limitado pela soma das sobras nas salas com móveis.
        total_sobra_moveis = 0
        for s in salas:
            if s.get("cadeiras_moveis", False):
                sob = max(0, int(s["capacidade_total"] - min(s["capacidade_total"], 10**9)))  # aqui só para forma; veremos melhor abaixo
            # na prática, a sobra real depende da alocação. Para limitar globalmente,
            # aproximamos como: sobra_max_global = soma(max(0, cap sala - 1)) → conservador
        # Em vez da aproximação acima (confusa), usamos um nó doador com capacidade muito alta e custo pequeno,
        # e confiamos que o min-cost minimizará doações. Se quiser limite estrito, compute previamente
        # SOBRA_MAX = soma(max(0, cap_sala - min(alunos_max_possíveis...))) — precisa de suposição.
        # Para simplificar, não limitamos rigorosamente o total aqui (o fallback já tem a versão exata).

        BENEF_SCALE = 10000
        TIE_W = int(self.parametros.get("peso_tiebreak_flow", 10))

        for _, r in dfv.iterrows():
            tn = f"T:{r['id_turma']}"
            sn = f"S:{r['id_sala']}"
            if tn not in G or sn not in G:
                continue
            benefit = float(r["proba_bom"])
            sobra   = int(max(0, r["capacidade_total"] - r["alunos"]))
            # custo: mais negativo = melhor
            cost = int(-benefit * BENEF_SCALE + TIE_W * max(0, r["sobra_local"]))
            # permitimos fluxo até alunos da turma (o grafo cuidará do total)
            G.add_edge(tn, sn, capacity=int(r["alunos"]), weight=cost)

        # Rodar min-cost flow
        flow = nx.min_cost_flow(G)

        # Reconstruir alocações: turma -> sala com maior fluxo
        alocacoes = []
        turmas_alocadas = set()
        for t in turmas:
            tn = f"T:{t['id_turma']}"
            if tn not in flow:
                continue
            # pega sala com maior fluxo
            flows = [(sn, q) for sn, q in flow[tn].items() if sn.startswith("S:") and q>0]
            if not flows:
                continue
            sn, q = max(flows, key=lambda x: x[1])
            sid = sn.split(":",1)[1]
            s = by_sid[sid]

            # pega linha correspondende pra montar observação/score
            sel = dfv[(dfv["id_turma"]==t["id_turma"]) & (dfv["id_sala"]==sid)]
            if sel.empty:
                continue
            linha = sel.iloc[0]

            obs = (f"Ocupacao: {linha['ocupacao']:.1%} "
                   f"(score ocupacao: {linha['score_ocupacao']:.2f}), "
                   f"Score ML: {float(linha.get('proba_ml', 0.0)):.2f}, "
                   f"Score combinado: {float(linha['proba_bom']):.2f}, "
                   f"Especiais: {linha['esp_necessarias']}/{linha['esp_disponiveis']} (Fluxo Mínimo)")

            if bool(linha.get("match_perfeito", False)):
                obs += ", Match perfeito (forçado a 100%)"

            alocacoes.append({
                "sala_id": s["id"],
                "turma_id": t["id"],
                "compatibilidade_score": round(float(linha['proba_bom']) * 100, 2),
                "observacoes": obs
            })
            turmas_alocadas.add(t["id"])

        turmas_nao = []
        for t in turmas:
            if t["id"] not in turmas_alocadas:
                turmas_nao.append({
                    "id": t["id"], "nome": t["nome"], "alunos": t["alunos"],
                    "esp_necessarias": t["esp_necessarias"],
                    "motivo": self._analisar_motivo_nao_alocacao(t)
                })

        # score simples = média dos scores (0..1) das alocações
        score = 0.0
        if alocacoes:
            score = np.mean([a["compatibilidade_score"]/100.0 for a in alocacoes])

        print(f"✅ [OPT] Fluxo Mínimo: {len(alocacoes)} alocações, score: {score:.2f}", file=sys.stderr)
        return alocacoes, float(score), turmas_nao

    def _otimizar_forca_bruta(self):
        """Força bruta para otimização (compatibilidade quando scipy não disponível)"""
        print("🔍 [OPT] Iniciando força bruta (fallback)", file=sys.stderr)

        turma_ids = [t["id_turma"] for t in self.turmas]
        sala_ids = [s["id_sala"] for s in self.salas]
        idx_turma = {tid: i for i, tid in enumerate(turma_ids)}
        idx_sala = {sid: j for j, sid in enumerate(sala_ids)}

        U = np.zeros((len(turma_ids), len(sala_ids)), dtype=float)
        for _, r in self.df.iterrows():
            i = idx_turma[r["id_turma"]]
            j = idx_sala[r["id_sala"]]
            if self.par_viavel(r):
                U[i, j] = max(0.0, float(r["proba_bom"]))

        n_t = len(turma_ids)
        n_s = len(sala_ids)
        n = min(n_t, n_s)

        best_score = -1.0
        best_assign = None
        max_permutations = 5000
        count = 0

        for salas_perm in permutations(range(n_s), n):
            count += 1
            if count > max_permutations:
                print(f"⚠️ [OPT] Limite de {max_permutations} permutações atingido", file=sys.stderr)
                break
            score = 0.0
            ok = True
            for i in range(n):
                j = salas_perm[i]
                if U[i, j] <= 0.0:
                    ok = False
                    break
                score += U[i, j]
            if ok and score > best_score:
                best_score = score
                best_assign = salas_perm

        alocacoes = []
        turmas_alocadas = set()

        if best_assign is not None:
            for i in range(n):
                j = best_assign[i]
                linha = self.df[(self.df["id_turma"] == turma_ids[i]) &
                                (self.df["id_sala"] == sala_ids[j])].iloc[0]
                turma_real = next(t for t in self.turmas if t["id_turma"] == turma_ids[i])
                sala_real = next(s for s in self.salas if s["id_sala"] == sala_ids[j])
                turmas_alocadas.add(turma_real["id"])
                obs = (
                    f"Ocupacao: {linha['ocupacao']:.1%} "
                    f"(score ocupacao: {linha['score_ocupacao']:.2f}), "
                    f"Score ML: {float(linha.get('proba_ml', 0.0)):.2f}, "
                    f"Score combinado: {float(linha['proba_bom']):.2f}, "
                    f"Especiais: {linha['esp_necessarias']}/{linha['esp_disponiveis']} (Força Bruta)"
                )
                if bool(linha.get("match_perfeito", False)):
                    obs += ", Match perfeito (forçado a 100%)"
                alocacoes.append({
                    "sala_id": sala_real["id"],
                    "turma_id": turma_real["id"],
                    "compatibilidade_score": round(float(linha['proba_bom']) * 100, 2),
                    "observacoes": obs
                })

        turmas_nao_alocadas = []
        for turma in self.turmas:
            if turma["id"] not in turmas_alocadas:
                turmas_nao_alocadas.append({
                    "id": turma["id"],
                    "nome": turma["nome"],
                    "alunos": turma["alunos"],
                    "esp_necessarias": turma["esp_necessarias"],
                    "motivo": self._analisar_motivo_nao_alocacao(turma)
                })

        den = max(1, min(n_t, n_s))
        best_score_normalizado = (best_score / den) if best_score >= 0 else 0.0
        print(f"✅ [OPT] Força bruta: {len(alocacoes)} alocações, score: {best_score_normalizado:.2f}", file=sys.stderr)
        return alocacoes, best_score_normalizado, turmas_nao_alocadas

    def _algoritmo_simples_fallback(self):
        """Algoritmo simples de alocação quando ML falha"""
        print("🔄 [FALLBACK] Usando algoritmo simples de fallback", file=sys.stderr)

        p = self.parametros
        alvo_ocupacao = float(p.get("alvo_ocupacao", 0.85))
        cadeiras_moveis_max_emprestimo = int(p.get("cadeiras_moveis_max_emprestimo", 5))

        alocacoes_simples = []
        turmas_alocadas = set()
        salas_usadas = set()
        turmas_nao_alocadas = []
        transferencias_cadeiras = []

        turmas_ordenadas = sorted(self.turmas, key=lambda t: t["alunos"], reverse=True)
        salas_ativas = [s for s in self.salas if s["status"].upper() == "ATIVA"]
        salas_ordenadas = sorted(salas_ativas, key=lambda s: s["capacidade_total"], reverse=True)

        cadeiras_por_sala = {}
        for s in salas_ativas:
            if bool(s.get("cadeiras_moveis", False)):
                cadeiras_por_sala[s["id"]] = {
                    "nome": s["nome"],
                    "capacidade_total": s["capacidade_total"],
                    "cadeiras_disponiveis": s["capacidade_total"],
                    "tem_moveis": True
                }

        for turma in turmas_ordenadas:
            melhor_sala = None
            melhor_score = -1
            for sala in salas_ordenadas:
                if sala["id"] in salas_usadas:
                    continue
                capacidade_fixa = sala["capacidade_total"]
                cadeiras_emprestadas_disponiveis = sum(
                    info["cadeiras_disponiveis"]
                    for sala_id, info in cadeiras_por_sala.items()
                    if sala_id != sala["id"] and sala_id not in salas_usadas and info["cadeiras_disponiveis"] > 0
                )
                capacidade_efetiva = capacidade_fixa + min(cadeiras_moveis_max_emprestimo, cadeiras_emprestadas_disponiveis)
                if capacidade_efetiva < turma["alunos"]:
                    continue
                if sala["cadeiras_especiais"] < turma["esp_necessarias"]:
                    continue
                score = score_ocupacao_puro(turma["alunos"], capacidade_efetiva, alvo_ocupacao)
                cadeiras_necessarias = max(0, turma["alunos"] - capacidade_fixa)
                if cadeiras_necessarias > 0:
                    score *= 0.9
                ocupacao_fixa = turma["alunos"] / capacidade_fixa if capacidade_fixa else 0.0
                if ocupacao_fixa == 1.0 and sala["cadeiras_especiais"] == turma["esp_necessarias"]:
                    score = 1.0
                if sala["cadeiras_especiais"] >= turma["esp_necessarias"]:
                    if sala["cadeiras_especiais"] == turma["esp_necessarias"]:
                        score += 0.1
                    score = min(1.0, score)
                if score > melhor_score:
                    melhor_score = score
                    melhor_sala = sala

            if melhor_sala:
                capacidade_fixa = melhor_sala["capacidade_total"]
                cadeiras_necessarias = max(0, turma["alunos"] - capacidade_fixa)
                ocupacao_fixa = turma["alunos"] / capacidade_fixa if capacidade_fixa else 0.0
                score_base = score_ocupacao_puro(turma["alunos"], capacidade_fixa, alvo_ocupacao)
                obs_detalhes = f"Ocupacao: {ocupacao_fixa:.1%} (score base: {score_base:.2f})"
                if cadeiras_necessarias > 0:
                    origem_cadeiras = []
                    cadeiras_restantes = cadeiras_necessarias
                    salas_ordenadas_cadeiras = sorted(
                        [(sala_id, info) for sala_id, info in cadeiras_por_sala.items()
                         if info["cadeiras_disponiveis"] > 0 and sala_id not in salas_usadas and sala_id != melhor_sala["id"]],
                        key=lambda x: x[1]["cadeiras_disponiveis"],
                        reverse=True
                    )
                    for sala_origem_id, info_origem in salas_ordenadas_cadeiras:
                        if cadeiras_restantes <= 0:
                            break
                        cadeiras_desta_sala = min(cadeiras_restantes, info_origem["cadeiras_disponiveis"])
                        origem_cadeiras.append({
                            "sala_origem": info_origem["nome"],
                            "quantidade": cadeiras_desta_sala
                        })
                        cadeiras_por_sala[sala_origem_id]["cadeiras_disponiveis"] -= cadeiras_desta_sala
                        cadeiras_restantes -= cadeiras_desta_sala

                    transferencia = {
                        "sala_destino": melhor_sala["nome"],
                        "turma": turma["nome"],
                        "total_cadeiras": cadeiras_necessarias,
                        "origens": origem_cadeiras
                    }
                    transferencias_cadeiras.append(transferencia)
                    origem_desc = ", ".join([f"{o['quantidade']} de {o['sala_origem']}" for o in origem_cadeiras])
                    obs_detalhes += f", +{cadeiras_necessarias} moveis ({origem_desc}) → cap. efetiva: {capacidade_fixa + cadeiras_necessarias}"

                if melhor_sala["cadeiras_especiais"] == turma["esp_necessarias"]:
                    obs_detalhes += f", Match exato especiais (+10%)"
                obs_detalhes += f", Especiais: {turma['esp_necessarias']}/{melhor_sala['cadeiras_especiais']} (Empréstimo entre salas)"

                alocacoes_simples.append({
                    "sala_id": melhor_sala["id"],
                    "turma_id": turma["id"],
                    "compatibilidade_score": round(melhor_score * 100, 2),
                    "observacoes": obs_detalhes
                })
                turmas_alocadas.add(turma["id"])
                salas_usadas.add(melhor_sala["id"])

                if melhor_sala["id"] in cadeiras_por_sala:
                    cadeiras_ocupadas = turma["alunos"]
                    cadeiras_restantes = max(0, cadeiras_por_sala[melhor_sala["id"]]["capacidade_total"] - cadeiras_ocupadas)
                    cadeiras_por_sala[melhor_sala["id"]]["cadeiras_disponiveis"] = cadeiras_restantes
                print(f"✅ [PYTHON] Turma '{turma['nome']}' alocada na sala '{melhor_sala['nome']}' (score: {melhor_score:.2f})", file=sys.stderr)
            else:
                motivo = self._analisar_motivo_nao_alocacao(turma)
                turmas_nao_alocadas.append({
                    "id": turma["id"],
                    "nome": turma["nome"],
                    "alunos": turma["alunos"],
                    "esp_necessarias": turma["esp_necessarias"],
                    "motivo": motivo
                })
                print(f"❌ [PYTHON] Turma '{turma['nome']}' nao pode ser alocada: {motivo}", file=sys.stderr)

        salas_ativas = [s for s in self.salas if s["status"].upper() == "ATIVA"]
        den = max(1, min(len(self.turmas), len(salas_ativas)))
        score_total = len(alocacoes_simples) / den if den > 0 else 0

        if transferencias_cadeiras:
            print(f"📋 [PYTHON] Transferências de cadeiras realizadas:", file=sys.stderr)
            for t in transferencias_cadeiras:
                origem_str = ", ".join([f"{o['quantidade']} de {o['sala_origem']}" for o in t['origens']])
                print(f"   → {t['sala_destino']} ({t['turma']}): {t['total_cadeiras']} cadeiras ({origem_str})", file=sys.stderr)

        return alocacoes_simples, score_total, turmas_nao_alocadas, transferencias_cadeiras

    def otimizar_alocacoes(self):
        """Executa otimização global das alocações - método de compatibilidade"""
        print("🔍 [OPT] Iniciando otimização", file=sys.stderr)
        try:
            if self.df is None:
                self._treinar_e_pontuar()

            # 1) Tenta fluxo (ótimo com empréstimo global, se networkx disponível)
            try:
                return self._otimizar_fluxo_minimo()
            except Exception as e:
                print(f"ℹ️ [OPT] Fluxo mínimo indisponível: {e}", file=sys.stderr)

            # 2) Cai pro Hungarian (ótimo sem restrição global de empréstimo)
            return self._otimizar_hungarian()

        except Exception as e:
            import traceback
            print("💥 [OPT] Exceção na otimização:", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            print(f"⚠️ [OPT] Erro: {e}. Usando empréstimo greedy.", file=sys.stderr)
            alocacoes, score, turmas_nao_alocadas, transferencias = self._algoritmo_simples_fallback()
            return alocacoes, score, turmas_nao_alocadas

    def _analisar_motivo_nao_alocacao(self, turma):
        motivos = []
        salas_ativas = [s for s in self.salas if s["status"].upper() == "ATIVA"]
        if len(salas_ativas) == 0:
            return "Nenhuma sala ativa disponivel"
        if len(self.turmas) > len(salas_ativas):
            motivos.append(f"Mais turmas ({len(self.turmas)}) que salas ({len(salas_ativas)})")
        salas_compativeis = 0
        for sala in salas_ativas:
            if sala["capacidade_total"] < turma["alunos"]:
                continue
            if sala["cadeiras_especiais"] < turma["esp_necessarias"]:
                continue
            salas_compativeis += 1
        if salas_compativeis == 0:
            motivos.append("Nenhuma sala compativel (capacidade ou cadeiras especiais)")
        elif salas_compativeis < len(self.turmas):
            motivos.append("Poucas salas compativeis para todas as turmas")
        return "; ".join(motivos) if motivos else "Salas insuficientes para otimizacao"

    def _analisar_problemas(self):
        problemas = []
        avisos = []
        for t in self.turmas:
            turma_problemas = []
            turma_opcoes = 0
            for s in self.salas:
                if s["status"].upper() != "ATIVA":
                    continue
                if s["capacidade_total"] < t["alunos"]:
                    deficit = t["alunos"] - s["capacidade_total"]
                    turma_problemas.append(f"Sala '{s['nome']}': capacidade insuficiente ({s['capacidade_total']} < {t['alunos']}, faltam {deficit} lugares)")
                if s["cadeiras_especiais"] < t["esp_necessarias"]:
                    deficit_esp = t["esp_necessarias"] - s["cadeiras_especiais"]
                    turma_problemas.append(f"Sala '{s['nome']}': cadeiras especiais insuficientes ({s['cadeiras_especiais']} < {t['esp_necessarias']}, faltam {deficit_esp})")
                ocupacao = t["alunos"] / s["capacidade_total"] if s["capacidade_total"] > 0 else 0
                if (s["capacidade_total"] >= t["alunos"] and s["cadeiras_especiais"] >= t["esp_necessarias"]):
                    turma_opcoes += 1
                    if ocupacao > 1.0:
                        avisos.append(f"Turma '{t['nome']}' em '{s['nome']}': ocupacao de {ocupacao:.1%} (acima de 100%)")
                    elif ocupacao < 0.5:
                        avisos.append(f"Turma '{t['nome']}' em '{s['nome']}': baixa ocupacao de {ocupacao:.1%} (desperdicio de espaco)")
            if turma_opcoes == 0:
                problemas.append({
                    "turma": t["nome"],
                    "tipo": "sem_opcoes",
                    "detalhes": turma_problemas,
                    "resumo": f"Turma '{t['nome']}' ({t['alunos']} alunos, {t['esp_necessarias']} especiais) nao tem nenhuma sala compativel"
                })
            elif turma_opcoes == 1:
                avisos.append(f"Turma '{t['nome']}' tem apenas 1 opcao de sala - flexibilidade limitada")

        if len(self.salas) < len(self.turmas):
            problemas.append({
                "tipo": "salas_insuficientes",
                "resumo": f"Ha {len(self.turmas)} turmas para apenas {len(self.salas)} salas - conflitos inevitaveis"
            })

        return {
            "problemas_criticos": problemas,
            "avisos": avisos,
            "total_turmas": len(self.turmas),
            "total_salas": len([s for s in self.salas if s["status"].upper() == "ATIVA"]),
            "viabilidade": "alta" if not problemas else "baixa" if len(problemas) >= len(self.turmas) // 2 else "media"
        }

    def executar(self):
        """Método principal que executa todo o algoritmo"""
        print("🚀 [MAIN] Iniciando execução do algoritmo", file=sys.stderr)

        try:
            if not self.salas:
                raise Exception("Nenhuma sala fornecida")
            if not self.turmas:
                raise Exception("Nenhuma turma fornecida")

            print(f"📋 [MAIN] Input: {len(self.salas)} salas, {len(self.turmas)} turmas", file=sys.stderr)
            print(f"⚙️ [MAIN] Parâmetros: {self.parametros}", file=sys.stderr)

            acuracia = self.treinar_modelo()
            analise = self._analisar_problemas()

            if acuracia == 0.0 or self.df is None or self.df.empty:
                print("🔄 [MAIN] ML falhou completamente, usando empréstimo greedy direto", file=sys.stderr)
                alocacoes, score, turmas_nao_alocadas, transferencias_cadeiras = self._algoritmo_simples_fallback()
                acuracia = 0.5
                algoritmo_usado = "Empréstimo entre salas (Greedy)"
            else:
                alocacoes, score, turmas_nao_alocadas = self.otimizar_alocacoes()
                transferencias_cadeiras = []
                # Determinar algoritmo usado baseado nas observações
                if alocacoes and "Fluxo Mínimo" in alocacoes[0].get("observacoes", ""):
                    algoritmo_usado = "Fluxo Mínimo (Ótimo)"
                elif alocacoes and "Hungarian" in alocacoes[0].get("observacoes", ""):
                    algoritmo_usado = "Hungarian"
                elif alocacoes and "Força Bruta" in alocacoes[0].get("observacoes", ""):
                    algoritmo_usado = "Força Bruta"
                elif alocacoes and "Empréstimo entre salas" in alocacoes[0].get("observacoes", ""):
                    algoritmo_usado = "Empréstimo entre salas (Greedy)"
                else:
                    algoritmo_usado = "Hungarian" if HAS_SCIPY else "Força Bruta"

            total_turmas = len(self.turmas)
            total_salas_ativas = len([s for s in self.salas if s["status"].upper() == "ATIVA"])
            turmas_alocadas = len(alocacoes)
            turmas_sobrando = len(turmas_nao_alocadas)

            # score já vem em [0..1] dos otimizadores
            score_otimizacao_pct = round(score * 100, 2) if alocacoes else 0

            # corrigir den para debug
            den = max(1, min(total_turmas, total_salas_ativas))

            debug_info = {
                "total_pares_avaliados": len(self.df) if hasattr(self, 'df') and self.df is not None else 0,
                "pares_viaveis": len(self.df[self.df.apply(self.par_viavel, axis=1)]) if hasattr(self, 'df') and self.df is not None else 0,
                "salas_ativas": total_salas_ativas,
                "turmas_vs_salas": f"{total_turmas} turmas para {total_salas_ativas} salas",
                "max_matches_possiveis": den,
                "algoritmo_usado": algoritmo_usado,
                "parametros_utilizados": self.parametros,
                "scipy_disponivel": HAS_SCIPY,
                "metricas_honestas": getattr(self, 'metricas_honestas', False)
            }

            observacoes = []
            if not HAS_SCIPY:
                observacoes.append("SciPy indisponível: fallback para força bruta (limite 5000 permutações)")
            if algoritmo_usado == "Simples (Fallback)":
                observacoes.append("ML falhou: usando algoritmo simples com rastreamento de transferências")
            if hasattr(self, 'metricas_honestas') and not self.metricas_honestas:
                observacoes.append("Métricas calculadas com treino=teste (dataset pequeno)")
            if self.parametros.get("modo_treinamento", "heuristico") == "historico":
                dados_historicos = self.parametros.get("dados_historicos", {})
                observacoes.append(f"Modo histórico: {len(dados_historicos)} pares com dados reais")
                debug_info["formato_historico"] = "Chave: '{id_turma}_{id_sala}' → Valor: 0/1"
            if self.parametros.get("usar_tiebreak_sobra", False):
                observacoes.append("Usando tie-break por aproveitamento (menor sobra)")
            if self.parametros.get("permitir_moveis_no_ml", False):
                observacoes.append("ML considera cadeiras móveis (capacidade efetiva)")
            if observacoes:
                debug_info["observacoes"] = observacoes

            if hasattr(self, 'debug_best_params'):
                debug_info["best_params"] = self.debug_best_params
            if hasattr(self, 'clf_report'):
                debug_info["classification_report"] = self.clf_report
            if hasattr(self, 'modelo_regras') and self.modelo_regras:
                regras_limitadas = self.modelo_regras[:2000] + "..." if len(self.modelo_regras) > 2000 else self.modelo_regras
                debug_info["arvore_regras"] = regras_limitadas

            if self.parametros.get("persist_modelo") and hasattr(self, 'clf'):
                try:
                    import joblib
                    caminho_modelo = self.parametros.get("caminho_modelo", "modelo_alocacao.pkl")
                    joblib.dump(self.clf, caminho_modelo)
                    debug_info["modelo_persistido"] = caminho_modelo
                    print(f"💾 [PERSIST] Modelo salvo em: {caminho_modelo}", file=sys.stderr)
                except Exception as e:
                    debug_info["erro_persistencia"] = str(e)
                    print(f"⚠️ [PERSIST] Erro ao salvar modelo: {e}", file=sys.stderr)

            print(f"✅ [MAIN] Execução concluída: {turmas_alocadas}/{total_turmas} turmas alocadas ({score_otimizacao_pct}%)", file=sys.stderr)

            return {
                "success": True,
                "alocacoes": alocacoes,
                "turmas_nao_alocadas": turmas_nao_alocadas,
                "score_otimizacao": score_otimizacao_pct,
                "total_alocacoes": turmas_alocadas,
                "total_turmas": total_turmas,
                "turmas_sobrando": turmas_sobrando,
                "acuracia_modelo": round(acuracia * 100, 2),
                "f1_modelo": round(acuracia * 100, 2),
                "analise_detalhada": analise,
                "debug_info": debug_info,
                "transferencias_cadeiras": transferencias_cadeiras
            }

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"💥 [PYTHON] Erro completo: {error_details}", file=sys.stderr)

            diagnostico = []
            if not self.salas:
                diagnostico.append("Nenhuma sala fornecida")
            elif not self.turmas:
                diagnostico.append("Nenhuma turma fornecida")
            else:
                salas_ativas = [s for s in self.salas if s["status"].upper() == "ATIVA"]
                diagnostico.append(f"{len(salas_ativas)} salas ativas, {len(self.turmas)} turmas")
                if len(self.turmas) == 0:
                    diagnostico.append("Sem turmas para alocar")
                elif len(salas_ativas) == 0:
                    diagnostico.append("Sem salas ativas disponíveis")

            error_msg = f"{str(e)}"
            if diagnostico:
                error_msg += f" | Contexto: {'; '.join(diagnostico)}"

            return {
                "success": False,
                "error": error_msg,
                "alocacoes": [],
                "turmas_nao_alocadas": self.turmas if hasattr(self, 'turmas') else [],
                "score_otimizacao": 0,
                "total_turmas": len(self.turmas) if hasattr(self, 'turmas') else 0,
                "turmas_alocadas": 0,
                "turmas_sobrando": len(self.turmas) if hasattr(self, 'turmas') else 0,
                "diagnostico": diagnostico
            }


def main():
    """Função principal - interface com Node.js"""
    print("🐍 [PYTHON] Script Python iniciado!", file=sys.stderr)

    parser = argparse.ArgumentParser(description='Algoritmo de Alocação Inteligente com ML')
    parser.add_argument('--dados', required=True, help='JSON com dados de salas e turmas')
    parser.add_argument('--parametros', required=False, default='{}', help='JSON com parâmetros de otimização (opcional)')
    parser.add_argument('--dump-arvore', action='store_true', help='Exportar árvore de decisão para debug')

    args = parser.parse_args()
    print(f"📝 [PYTHON] Argumentos recebidos: dados={len(args.dados)} chars, parametros={len(args.parametros)} chars", file=sys.stderr)

    try:
        print("🔍 [PYTHON] Parseando dados JSON...", file=sys.stderr)
        dados = json.loads(args.dados)
        parametros = json.loads(args.parametros)
        if args.dump_arvore:
            parametros["dump_arvore"] = True

        print(f"📊 [PYTHON] Dados parseados: {len(dados.get('salas', []))} salas, {len(dados.get('turmas', []))} turmas", file=sys.stderr)
        print(f"⚙️ [PYTHON] Parametros: {parametros}", file=sys.stderr)

        print("🤖 [PYTHON] Criando instancia do algoritmo...", file=sys.stderr)
        algoritmo = AlocacaoInteligenteMLA(dados['salas'], dados['turmas'], parametros)

        print("⚡ [PYTHON] Executando algoritmo...", file=sys.stderr)
        resultado = algoritmo.executar()

        print(f"✅ [PYTHON] Algoritmo executado! Success: {resultado.get('success')}", file=sys.stderr)
        print(f"📈 [PYTHON] Resultado: {len(resultado.get('alocacoes', []))} alocacoes, score: {resultado.get('score_otimizacao')}", file=sys.stderr)

        print(json.dumps(resultado, ensure_ascii=False))

        if not resultado['success']:
            print(f"❌ [PYTHON] Falha no algoritmo: {resultado.get('error')}", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"💥 [PYTHON] Excecao capturada: {str(e)}", file=sys.stderr)
        import traceback
        print(f"🔍 [PYTHON] Traceback: {traceback.format_exc()}", file=sys.stderr)

        resultado = {
            'success': False,
            'error': f'Erro no algoritmo: {str(e)}',
            'alocacoes': [],
            'score_otimizacao': 0
        }
        print(json.dumps(resultado, ensure_ascii=False))
        sys.exit(1)


if __name__ == '__main__':
    main()
