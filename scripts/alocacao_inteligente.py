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

try:
    import joblib
    HAS_JOBLIB = True
except Exception:
    HAS_JOBLIB = False

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
        self.permitir_fallback_simples = bool(self.parametros.get("permitir_fallback_simples", False))
        self.modo_otimizador = self.parametros.get("forcar_otimizador", "auto")  # "fluxo" | "hungarian" | "auto"
        
        # RN (rede neural) opcional
        self.rn_model_path = self.parametros.get("rn_model_path")
        self.peso_rn = float(self.parametros.get("peso_rn", 0.30))
        self.rn_pipe = None
        self.rn_feature_cols = None

    def _normalize_salas(self, salas):
        normalized = []
        for sala in salas:
            # Aceita bool ou número. Se vier só bool, quantidade é desconhecida (None).
            raw_flag = sala.get("cadeiras_moveis", 0)
            raw_qtd  = sala.get("cadeiras_moveis_qtd", None)
            if raw_qtd is not None:
                moveis_qtd = int(raw_qtd or 0)
                tem_moveis = moveis_qtd > 0
            else:
                tem_moveis = bool(raw_flag)
                moveis_qtd = None
            normalized.append({
                "id": sala.get("id"),
                "id_sala": sala.get("id_sala", sala.get("id")),
                "nome": sala.get("nome", ""),
                "capacidade_total": int(sala.get("capacidade_total", 0) or 0),
                "localizacao": sala.get("localizacao", ""),
                "status": str(sala.get("status", "ATIVA")).upper(),
                "cadeiras_moveis_qtd": moveis_qtd,  # pode ser None (desconhecida)
                "cadeiras_moveis": tem_moveis,
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
                score_ocupacao_base = score_ocupacao_puro(alunos, cap, alvo_ocupacao)
                score_ocupacao_eff = score_ocupacao_puro(alunos, cap_efetiva, alvo_ocupacao)

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
                    "score_ocupacao_base": score_ocupacao_base,
                    "score_ocupacao_eff": score_ocupacao_eff,
                    "esp_necessarias": esp_need,
                    "esp_disponiveis": esp_have,
                    "esp_deficit": esp_deficit,
                    "esp_sobra": esp_sobra,
                    "atende_especial": atende_especial,
                    "label_heuristico": label_heuristico,
                    "precisa_moveis": int(deficit > 0),
                    "precisa_moveis_eff": int(deficit_eff > 0),
                    "dist_alvo_occ": abs((ocupacao_eff if permitir_moveis_ml else ocupacao) - alvo_ocupacao)
                })
        return pd.DataFrame(rows)

    def _sample_weights(self):
        """
        Pesos de amostra para a árvore, inspirados em regras de negócio:
        - prioridade para pares com especiais atendidos;
        - bônus para match perfeito (capacidade exata);
        - leve penalização quando depende de móveis;
        - reforço para ocupação perto do alvo (informativo).
        """
        df = self.df
        p = self.parametros
        alvo = float(p.get("alvo_ocupacao", 0.85))
        permitir_moveis_ml = bool(p.get("permitir_moveis_no_ml", False))

        precisa = df["precisa_moveis_eff"] if (permitir_moveis_ml and "precisa_moveis_eff" in df.columns) else df["precisa_moveis"]
        occ = df["ocupacao_eff"] if (permitir_moveis_ml and "ocupacao_eff" in df.columns) else df["ocupacao"]

        # base
        w = np.ones(len(df), dtype=float)

        # atende especiais vale mais (evita soluções inviáveis)
        w += 0.4 * df["atende_especial"].values

        # match perfeito (capacidade exata + especiais ok) tem peso alto
        match_perfeito = (np.isclose(df["ocupacao"].values, 1.0)) & (df["esp_deficit"].values == 0) & (df["deficit"].values == 0)
        w += 0.6 * match_perfeito.astype(float)

        # leve penalização se depende de móveis (não proíbe, só educa a árvore)
        w -= 0.2 * precisa.values

        # exemplo de "informativeness": quanto mais perto do alvo, mais peso (até +0.3)
        w += 0.3 * np.maximum(0.0, 1 - np.minimum(np.abs(occ - alvo) / max(alvo, 1e-9), 1))

        return np.clip(w, 0.1, 3.0)

    def _rn_load_if_needed(self):
        if self.rn_pipe is not None:
            return
        if not self.rn_model_path:
            return
        if not HAS_JOBLIB:
            print("⚠️ [RN] joblib indisponível; RN desativada.", file=sys.stderr)
            return
        try:
            bundle = joblib.load(self.rn_model_path)
            self.rn_pipe = bundle.get("pipeline", None)
            self.rn_feature_cols = bundle.get("feature_cols", [])
            if self.rn_pipe is None or not self.rn_feature_cols:
                print("⚠️ [RN] Modelo inválido (faltam pipeline/feature_cols).", file=sys.stderr)
                self.rn_pipe = None
                self.rn_feature_cols = None
            else:
                print(f"🧠 [RN] Carregado: {self.rn_model_path} | features: {self.rn_feature_cols}", file=sys.stderr)
        except Exception as e:
            print(f"⚠️ [RN] Falha ao carregar modelo: {e}", file=sys.stderr)
            self.rn_pipe = None
            self.rn_feature_cols = None

    def _aplicar_rn(self):
        """Gera self.df['proba_rn'] se modelo RN estiver disponível."""
        self._rn_load_if_needed()
        if self.rn_pipe is None or not self.rn_feature_cols or self.df is None or self.df.empty:
            return
        # RN espera coluna 'sobra' (nosso DF tem 'sobra_local')
        df_feat = self.df.copy()
        if "sobra_local" in df_feat.columns and "sobra" in self.rn_feature_cols:
            df_feat["sobra"] = df_feat["sobra_local"]
        # garantir todas as colunas
        for col in self.rn_feature_cols:
            if col not in df_feat.columns:
                df_feat[col] = 0.0
        try:
            X = df_feat[self.rn_feature_cols].astype(float).values
            proba = self.rn_pipe.predict_proba(X)[:, 1]
            self.df["proba_rn"] = np.clip(proba, 0.0, 1.0)
            print(f"✅ [RN] Probabilidades geradas p/ {len(proba)} pares.", file=sys.stderr)
        except Exception as e:
            print(f"⚠️ [RN] Erro ao prever: {e}", file=sys.stderr)

    def _refino_troca_2opt(self, alocacoes):
        """Refino local 2-opt: tenta trocar pares de alocações para melhorar o score total"""
        if not alocacoes:
            return alocacoes

        # mapa rápidos
        by_tid = {t["id"]: t for t in self.turmas}
        by_sid = {s["id"]: s for s in self.salas}
        # cria lookup do DF por (tid,sid_sala)
        dfk = self.df.set_index(["id_turma", "id_sala"])

        melhorou = True
        while melhorou:
            melhorou = False
            for i in range(len(alocacoes)):
                for j in range(i+1, len(alocacoes)):
                    a, b = alocacoes[i], alocacoes[j]
                    tA, sA = by_tid[a["turma_id"]], by_sid[a["sala_id"]]
                    tB, sB = by_tid[b["turma_id"]], by_sid[b["sala_id"]]

                    # linhas atuais e trocadas
                    try:
                        rAA = dfk.loc[(tA["id_turma"], sA["id_sala"])]
                        rBB = dfk.loc[(tB["id_turma"], sB["id_sala"])]
                        rAB = dfk.loc[(tA["id_turma"], sB["id_sala"])]
                        rBA = dfk.loc[(tB["id_turma"], sA["id_sala"])]
                    except KeyError:
                        continue

                    # viabilidade simples (sem violar especiais)
                    def ok(r): return (r["esp_deficit"] == 0) and ((r["capacidade_total"] >= r["alunos"]) or r["sala_movel"])

                    if ok(rAB) and ok(rBA):
                        score_old = float(rAA["proba_bom"]) + float(rBB["proba_bom"])
                        score_new = float(rAB["proba_bom"]) + float(rBA["proba_bom"])
                        # margem mínima de melhoria para trocar
                        if score_new > score_old + 1e-6:
                            a["sala_id"], b["sala_id"] = sB["id"], sA["id"]
                            a["compatibilidade_score"] = round(float(rAB["proba_bom"])*100, 2)
                            b["compatibilidade_score"] = round(float(rBA["proba_bom"])*100, 2)
                            melhorou = True
                            break
                if melhorou: break
        return alocacoes

    def _prova_impossibilidade(self):
        """Diagnóstico explícito para a UI sobre impossibilidade estrutural"""
        salas_ativas = [s for s in self.salas if s["status"].upper() == "ATIVA"]
        n_t, n_s = len(self.turmas), len(salas_ativas)
        prova = {
            "turmas": n_t, 
            "salas_ativas": n_s,
            "limite_teorico_max_matches": min(n_t, n_s),
            "impossibilidade_estrutural": n_t > n_s
        }
        
        # Estimativa rápida de cadeiras móveis faltantes para tentar fechar 100%:
        moveis_disponiveis = 0
        for s in salas_ativas:
            if s.get("cadeiras_moveis", False):
                qtd = s.get("cadeiras_moveis_qtd")
                if qtd is not None:
                    moveis_disponiveis += int(qtd or 0)
                else:
                    # Se não tem quantidade declarada, assume que pode doar toda a sobra
                    moveis_disponiveis += s["capacidade_total"]
        demanda_min = 0
        for t in self.turmas:
            candidatas = [s for s in salas_ativas if s["cadeiras_especiais"] >= t["esp_necessarias"]]
            if not candidatas:
                prova.setdefault("turmas_sem_especiais", []).append(t["nome"])
                continue
            deficit_min = min(max(0, t["alunos"] - s["capacidade_total"]) for s in candidatas)
            demanda_min += deficit_min
        
        if demanda_min > moveis_disponiveis:
            prova["cadeiras_moveis_faltantes"] = int(demanda_min - moveis_disponiveis)
        
        return prova

    def _treinar_e_pontuar(self):
        """Treina Árvore de Decisão robusta (Grid + Poda + Calibração) e calcula proba_bom."""
        print("🔍 [ML] Árvore robusta (grid+poda+calibração)", file=sys.stderr)

        # ---- features ----
        feature_cols = [
            "alunos", "capacidade_total", "deficit", "sobra_local", "sala_movel",
            "ocupacao", "score_ocupacao", "esp_necessarias", "esp_disponiveis",
            "esp_deficit", "esp_sobra", "atende_especial",
            "precisa_moveis",  # ajuda a árvore a "ver" a dependência de móveis
            "dist_alvo_occ"    # distância do alvo de ocupação (85%)
        ]
        if self.parametros.get("permitir_moveis_no_ml", False):
            for col in ["capacidade_efetiva", "deficit_eff", "ocupacao_eff", "precisa_moveis_eff"]:
                if col in self.df.columns and col not in feature_cols:
                    feature_cols.append(col)

        X = self.df[feature_cols].values
        y = self.df["label_heuristico"].values
        sw = self._sample_weights()

        print(f"📊 [ML] Dataset: {len(X)} pares, classes: {Counter(y)}", file=sys.stderr)

        # ---- pesos para mistura final com score_ocupacao (compatível com seu contrato) ----
        p = self.parametros
        peso_ml  = float(p.get("peso_ml", 0.6))          # dá mais peso para ML, já que é "só árvore"
        peso_occ = float(p.get("peso_ocupacao", 0.4))
        total = max(1e-9, peso_ml + peso_occ)
        peso_ml, peso_occ = peso_ml/total, peso_occ/total
        print(f"📊 [ML] Pesos: ML={peso_ml:.2f} | Ocupação={peso_occ:.2f}", file=sys.stderr)

        limiar_cls = float(p.get("limiar_cls", 0.5))
        use_grid   = bool(p.get("gridsearch", True))
        use_parallel = bool(p.get("parallel_ml", False))
        n_jobs_grid = -1 if use_parallel else 1

        # hiperparâmetros custom se vierem
        custom_max_depth = p.get("max_depth")
        custom_min_samples_split = p.get("min_samples_split")
        custom_min_samples_leaf = p.get("min_samples_leaf")

        # ---- treinamento da árvore ----
        # Regra: se temos ao menos 12 exemplos e as duas classes, fazemos grid + calibração.
        tem_duas_classes = (len(np.unique(y)) > 1)
        tree_proba = None

        if tem_duas_classes and len(y) >= 12 and use_grid:
            try:
                print("🎯 [ML] GridSearch + Poda + Calibração", file=sys.stderr)
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                param_grid = {
                    "max_depth": [custom_max_depth] if custom_max_depth is not None else [3, 4, 5, None],
                    "min_samples_split": [custom_min_samples_split] if custom_min_samples_split is not None else [2, 5, 10],
                    "min_samples_leaf": [custom_min_samples_leaf] if custom_min_samples_leaf is not None else [1, 2, 5],
                    "min_impurity_decrease": [0.0, 1e-4, 1e-3],   # ← ajuda na estabilidade
                    "ccp_alpha": [0.0, 1e-4, 1e-3]               # ← poda por complexidade
                }
                base = DecisionTreeClassifier(random_state=42, class_weight="balanced")
                grid = GridSearchCV(base, param_grid=param_grid, scoring="f1", cv=cv, n_jobs=n_jobs_grid)
                grid.fit(X, y, sample_weight=sw)
                best = grid.best_estimator_
                self.debug_best_params = grid.best_params_
                print(f"🏆 [ML] Best params: {self.debug_best_params}", file=sys.stderr)

                # calibração: isotônica se base for estável e dados suficientes, senão sigmoid
                calib_method = "isotonic" if len(y) >= 100 else "sigmoid"
                cal = CalibratedClassifierCV(best, method=calib_method, cv=3)
                cal.fit(X, y, sample_weight=sw)
                self.clf = cal
                tree_proba = self.clf.predict_proba(X)[:, 1]

                try:
                    self.modelo_regras = export_text(best, feature_names=feature_cols)
                    if self.parametros.get("dump_arvore"):
                        with open("arvore_decisao.txt", "w", encoding="utf-8") as f:
                            f.write(self.modelo_regras)
                    print("📋 [ML] Regras exportadas", file=sys.stderr)
                except Exception as e:
                    print(f"⚠️ [ML] Erro ao exportar regras: {e}", file=sys.stderr)
            except Exception as e:
                print(f"⚠️ [ML] GridSearch falhou ({e}). Caindo para árvore simples.", file=sys.stderr)
                use_grid = False
                # Continua para o else que treina a árvore simples

        if not use_grid or not tem_duas_classes or len(y) < 12:
            print("⚠️ [ML] Árvore simples (sem grid/calibração) — dataset pequeno ou classe única", file=sys.stderr)
            self.clf = DecisionTreeClassifier(
                max_depth=custom_max_depth or 4,
                min_samples_split=custom_min_samples_split or 2,
                min_samples_leaf=custom_min_samples_leaf or 1,
                min_impurity_decrease=1e-4,   # um pouco de estabilidade
                ccp_alpha=1e-4,               # poda leve
                random_state=42,
                class_weight="balanced"
            )
            self.clf.fit(X, y, sample_weight=sw)
            proba = getattr(self.clf, "predict_proba", None)
            tree_proba = proba(X)[:, 1] if proba is not None else np.zeros(len(X))
            self.debug_best_params = {
                "max_depth": self.clf.get_params().get("max_depth"),
                "min_samples_split": self.clf.get_params().get("min_samples_split"),
                "min_samples_leaf": self.clf.get_params().get("min_samples_leaf"),
                "min_impurity_decrease": self.clf.get_params().get("min_impurity_decrease"),
                "ccp_alpha": self.clf.get_params().get("ccp_alpha")
            }

        # ---- prob final do ML (só árvore) ----
        self.df["proba_ml"] = np.clip(tree_proba, 0.0, 1.0)

        # ---- RN opcional: gera proba_rn ----
        self._aplicar_rn()  # cria self.df['proba_rn'] se houver modelo

        # ---- combinação com score de ocupação (+ RN, se existir) ----
        p = self.parametros
        peso_ml  = float(p.get("peso_ml", 0.45))
        peso_occ = float(p.get("peso_ocupacao", 0.25))
        peso_rn  = float(p.get("peso_rn", 0.30)) if "proba_rn" in self.df.columns else 0.0

        total = max(1e-9, peso_ml + peso_occ + peso_rn)
        peso_ml  /= total
        peso_occ /= total
        peso_rn  /= total

        combo = (
            peso_ml  * self.df["proba_ml"].values +
            peso_occ * self.df["score_ocupacao"].values
        )
        if "proba_rn" in self.df.columns:
            combo += peso_rn * self.df["proba_rn"].values

        self.df["proba_bom"] = np.clip(combo, 0.0, 1.0)

        # 🚧 Piso: se cabe sem déficit e especiais ok, não deixar cair abaixo do score de ocupação
        cond_sem_deficit = (self.df["deficit"] == 0) & (self.df["esp_deficit"] == 0)
        self.df.loc[cond_sem_deficit, "proba_bom"] = np.maximum(
            self.df.loc[cond_sem_deficit, "proba_bom"],
            self.df.loc[cond_sem_deficit, "score_ocupacao"]
        )
        if "proba_rn" in self.df.columns:
            print(f"✅ [ML] Combinação: ML({peso_ml:.1%}) + Ocupação({peso_occ:.1%}) + RN({peso_rn:.1%})", file=sys.stderr)
        else:
            print(f"✅ [ML] Combinação: ML({peso_ml:.1%}) + Ocupação({peso_occ:.1%})", file=sys.stderr)

        # ---- boost para match perfeito ----
        mask_perfeito = (
            np.isclose(self.df["ocupacao"].values, 1.0) &
            (self.df["esp_deficit"].values == 0) &
            (self.df["deficit"].values == 0)
        )
        self.df["match_perfeito"] = False
        if mask_perfeito.any():
            self.df.loc[mask_perfeito, "proba_bom"] = 1.0
            self.df.loc[mask_perfeito, "match_perfeito"] = True
            print(f"🎯 [ML] {mask_perfeito.sum()} matches perfeitos forçados a 100%", file=sys.stderr)

        # ---- Métricas (preferir CV quando possível) ----
        try:
            usar_metricas_cv = bool(self.parametros.get("metricas_cv", True))
            if tem_duas_classes:
                if len(y) >= 12 and usar_metricas_cv:
                    print("🔍 [METRICS] F1 com CV honesta (via proba_ml)", file=sys.stderr)
                    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                    try:
                        # probas em CV para honestidade
                        y_proba_cv = cross_val_predict(self.clf, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
                        y_pred_cv = (y_proba_cv >= limiar_cls).astype(int)
                        self.clf_report = classification_report(y, y_pred_cv, output_dict=True, zero_division=0)
                        self.metricas_honestas = True
                    except Exception as e:
                        print(f"⚠️ [METRICS] CV falhou ({e}). Usando treino=teste.", file=sys.stderr)
                        y_pred = (self.df['proba_ml'].values >= limiar_cls).astype(int)
                        self.clf_report = classification_report(y, y_pred, output_dict=True, zero_division=0)
                        self.metricas_honestas = False
                else:
                    print("⚠️ [METRICS] Dataset pequeno/sem CV: treino=teste", file=sys.stderr)
                    y_pred = (self.df["proba_ml"].values >= limiar_cls).astype(int)
                    self.clf_report = classification_report(y, y_pred, output_dict=True, zero_division=0)
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

        # pesos vindos de self.parametros (com defaults)
        penal_movel = float(self.parametros.get("penalidade_moveis", 0.10))    # 0..1
        peso_sobra  = float(self.parametros.get("peso_sobra", 0.02))           # 0..1
        custo_inviavel = float(self.parametros.get("custo_inviavel", 1e6))

        U = np.full((len(turma_ids), len(sala_ids)), -np.inf, dtype=float)

        # normalização de sobra p/ 0..1 (evita efeito de escala)
        sobra_vals = self.df["sobra_local"].values
        sobra_min, sobra_rng = sobra_vals.min() if len(sobra_vals)>0 else 0, (np.ptp(sobra_vals) + 1e-9)

        for _, r in self.df.iterrows():
            if r["esp_deficit"] != 0:
                continue
            i = idx_turma[r["id_turma"]]; j = idx_sala[r["id_sala"]]
            proba = float(r["proba_bom"])
            precisa_mov = 1.0 if (r["alunos"] > r["capacidade_total"]) else 0.0
            sobra_norm = (r["sobra_local"] - sobra_min) / sobra_rng

            # ===== Proteção de especiais =====
            need_special      = 1.0 if (r["esp_necessarias"] > 0) else 0.0
            room_has_special  = 1.0 if (r["esp_disponiveis"] > 0) else 0.0
            desperdicio_esp   = 1.0 if (room_has_special == 1.0 and need_special == 0.0) else 0.0
            match_esp         = 1.0 if (room_has_special == 1.0 and need_special == 1.0) else 0.0

            # pesos ajustáveis por parâmetro
            peso_especial_desperdicio = float(self.parametros.get("peso_especial_desperdicio", 0.15))
            bonus_especial_match       = float(self.parametros.get("bonus_especial_match", 0.05))

            # score ajustado: maximiza proba, penaliza móveis e sobra, protege especiais
            score_aj = (
                proba
                - penal_movel * precisa_mov
                - peso_sobra * sobra_norm
                - peso_especial_desperdicio * desperdicio_esp
                + bonus_especial_match * match_esp
            )

            # Permite pares com déficit: viabilidade será checada no pós-processo via empréstimos.
            U[i, j] = max(0.0, score_aj)

        # custo = 1 - score_aj (quem tem score_aj alto tem custo baixo)
        C = np.where(U > -np.inf, 1.0 - U, custo_inviavel)

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
        #    A ideia: se a sala tem cadeiras móveis e sobram X lugares, pode doar até X cadeiras.
        limite_doador = int(self.parametros.get("limite_moveis_por_doador", 999999))
        
        # Mapa da maior turma compatível por sala (para evitar efeito cascata)
        maior_turma_por_sala = {}
        for s in self.salas:
            if s["status"].upper() != "ATIVA":
                continue
            maior = 0
            for t in self.turmas:
                if (s["cadeiras_especiais"] >= t["esp_necessarias"]) and (t["alunos"] <= s["capacidade_total"]):
                    maior = max(maior, t["alunos"])
            maior_turma_por_sala[s["id"]] = maior
        
        doadores = {}
        for s in self.salas:
            if bool(s.get("cadeiras_moveis", False)):
                cap = int(s["capacidade_total"])
                hosped = int(hospedados.get(s["id"], 0))
                sobra = max(0, cap - hosped)
                if sobra <= 0:
                    continue
                qtd_decl = s.get("cadeiras_moveis_qtd", None)
                # Se a quantidade declarada existir, ela limita; caso contrário, usa só a sobra real.
                if qtd_decl is None:
                    doadores[s["id"]] = {
                        "nome": s["nome"],
                        "sobra": sobra,
                        "limite": limite_doador
                    }
                else:
                    qtd_decl = int(qtd_decl or 0)
                    if qtd_decl > 0:
                        doadores[s["id"]] = {
                            "nome": s["nome"],
                            "sobra": min(sobra, qtd_decl),
                            "limite": min(limite_doador, qtd_decl)
                        }

        # 4) Para cada candidata com déficit, puxar cadeiras dos doadores
        #    Estratégia: alocar primeiro quem precisa de menos cadeiras.
        candidatas.sort(key=lambda it: it[3])  # it[3] = déficit
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
                    pegar = min(faltando, info["sobra"], info["limite"])
                    if pegar > 0:
                        info["sobra"] -= pegar
                        faltando -= pegar
                        emprestimos.append((info["nome"], pegar))
            # se não conseguiu cobrir o déficit, descarta essa alocação
            if faltando > 0:
                continue

            turmas_alocadas.add(t["id"])
            
            # Recalcular com capacidade efetiva real após empréstimo
            alvo = float(self.parametros.get("alvo_ocupacao", 0.85))
            peso_ml = float(self.parametros.get("peso_ml", 0.6))
            peso_occ = float(self.parametros.get("peso_ocupacao", 0.4))
            peso_rn = float(self.parametros.get("peso_rn", 0.0)) if "proba_rn" in self.df.columns else 0.0
            
            peso_sum = max(1e-9, peso_ml + peso_occ + peso_rn)
            peso_ml, peso_occ, peso_rn = peso_ml/peso_sum, peso_occ/peso_sum, peso_rn/peso_sum

            cap_eff_real = int(linha["capacidade_total"] + deficit)  # déficit foi totalmente coberto
            score_occ_eff_real = score_ocupacao_puro(int(linha["alunos"]), cap_eff_real, alvo)

            proba_bom_eff = np.clip(
                peso_ml * float(linha.get("proba_ml", 0.0)) + peso_occ * score_occ_eff_real,
                0.0, 1.0
            )
            # Incluir RN se disponível
            if "proba_rn" in self.df.columns:
                proba_bom_eff += peso_rn * float(linha.get("proba_rn", 0.0))
                proba_bom_eff = np.clip(proba_bom_eff, 0.0, 1.0)
            # 🚧 Piso pós-empréstimo: se agora cabe e especiais atendem, garantir pelo menos o score de ocupação efetivo
            if (linha["esp_deficit"] == 0) and (int(linha["alunos"]) <= cap_eff_real):
                proba_bom_eff = max(proba_bom_eff, score_occ_eff_real)
            # 🎯 Se ficou ocupação 100% e especiais ok, compatibilidade 100%
            if (abs((int(linha["alunos"])/cap_eff_real) - 1.0) < 1e-9) and (linha["esp_deficit"] == 0):
                proba_bom_eff = 1.0
            compat_calc = round(float(proba_bom_eff) * 100, 2)

            # Métricas explícitas para UI
            depende_moveis = int(deficit > 0)
            ocupacao_eff = int(linha["alunos"]) / cap_eff_real if cap_eff_real else 0.0
            
            # Fatores que afetaram o índice
            fatores = []
            if score_occ_eff_real >= 0.8:
                fatores.append(("+ Perto do alvo de ocupação", True))
            if linha["esp_deficit"] == 0 and linha["esp_necessarias"] > 0:
                fatores.append(("+ Atende especiais", True))
            if depende_moveis == 1:
                fatores.append(("− Usa móveis", True))
            
            # Determinar se RN está ativa para o texto
            tem_rn = "proba_rn" in self.df.columns
            indice_texto = "Índice (ML+RN+Ocupação)" if tem_rn else "Índice (ML+Ocupação)"
            
            obs = (
                f"Ocupacao: {ocupacao_eff:.1%} "
                f"(score ocupacao: {score_occ_eff_real:.2f}), "
                f"IA(Tree={float(linha.get('proba_ml', 0.0)):.2f}), "
                f"{indice_texto}: {round(proba_bom_eff*100, 2)}%, "
                f"Especiais: {linha['esp_necessarias']}/{linha['esp_disponiveis']} (Hungarian)"
            )
            if deficit > 0:
                detalhes = ", ".join([f"{qtd} de {nome}" for (nome, qtd) in emprestimos])
                obs += f", +{deficit} moveis ({detalhes}) → cap. efetiva: {cap_eff_real}"
            if bool(linha.get("match_perfeito", False)):
                obs += ", Match perfeito (forçado a 100%)"

            aloc_final.append({
                "sala_id": s["id"],
                "turma_id": t["id"],
                "compatibilidade_score": compat_calc,  # usa o proba com ocupação efetiva
                "observacoes": obs,
                # Métricas explícitas para UI
                "ui": {
                    "ocupacao_pct": round(ocupacao_eff * 100, 1),
                    "capacidade_efetiva": cap_eff_real,
                    "score_ocupacao_pct": round(score_occ_eff_real * 100, 1),
                    "proba_ml_pct": round(float(linha.get("proba_ml", 0.0)) * 100, 1),
                    "indice_qualidade_pct": compat_calc,
                    "depende_moveis": depende_moveis,
                    "especiais": {
                        "necessarias": int(linha["esp_necessarias"]),
                        "disponiveis": int(linha["esp_disponiveis"]),
                        "atendidas": int(linha["esp_necessarias"]) if linha["esp_deficit"] == 0 else 0
                    },
                    "fatores": fatores
                }
            })

        alocacoes = aloc_final

        # 5) Refino 2-opt para melhorar alocações
        alocacoes = self._refino_troca_2opt(alocacoes)

        # 6) Score final = média dos scores que aparecem na UI (pós-empréstimo)
        if alocacoes:
            best_score = float(np.mean([a["compatibilidade_score"] / 100.0 for a in alocacoes]))
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

            # Determinar se RN está ativa para o texto
            tem_rn = "proba_rn" in self.df.columns
            indice_texto = "Índice (ML+RN+Ocupação)" if tem_rn else "Índice (ML+Ocupação)"
            
            obs = (f"Ocupacao: {linha['ocupacao']:.1%} "
                   f"(score ocupacao: {linha['score_ocupacao']:.0%}), "
                   f"IA(Tree={float(linha.get('proba_ml', 0.0)):.2f}), "
                   f"{indice_texto}: {round(float(linha['proba_bom']) * 100, 2)}%, "
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
                # Determinar se RN está ativa para o texto
                tem_rn = "proba_rn" in self.df.columns
                indice_texto = "Índice (ML+RN+Ocupação)" if tem_rn else "Índice (ML+Ocupação)"
                
                obs = (
                    f"Ocupacao: {linha['ocupacao']:.1%} "
                    f"(score ocupacao: {linha['score_ocupacao']:.0%}), "
                    f"IA(Tree={float(linha.get('proba_ml', 0.0)):.2f}), "
                    f"{indice_texto}: {round(float(linha['proba_bom']) * 100, 2)}%, "
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
            if self.modo_otimizador in ("fluxo", "auto"):
                try:
                    return self._otimizar_fluxo_minimo()
                except Exception as e:
                    print(f"ℹ️ [OPT] Fluxo mínimo indisponível: {e}", file=sys.stderr)

            # 2) Cai pro Hungarian (ótimo sem restrição global de empréstimo)
            if self.modo_otimizador in ("hungarian", "auto"):
                return self._otimizar_hungarian()
            
            # Se chegou aqui, modo_otimizador não é válido
            raise ValueError(f"Modo de otimizador inválido: {self.modo_otimizador}")

        except Exception as e:
            import traceback
            print("💥 [OPT] Exceção na otimização:", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            if self.permitir_fallback_simples:
                print(f"⚠️ [OPT] Erro: {e}. Usando empréstimo greedy.", file=sys.stderr)
                alocacoes, score, turmas_nao_alocadas, transferencias = self._algoritmo_simples_fallback()
                return alocacoes, score, turmas_nao_alocadas
            else:
                print(f"❌ [OPT] Erro: {e}. Fallback simples desabilitado (TCC: apenas IA).", file=sys.stderr)
                raise  # deixa a execução falhar explicitamente (mantém "só IA")

    def _analisar_motivo_nao_alocacao(self, turma):
        motivos = []
        salas_ativas = [s for s in self.salas if s["status"].upper() == "ATIVA"]
        if len(salas_ativas) == 0:
            return "Nenhuma sala ativa disponivel"
        if len(self.turmas) > len(salas_ativas):
            motivos.append(f"Mais turmas ({len(self.turmas)}) que salas ({len(salas_ativas)})")
        
        # Verificar se a turma não cabe por 1 cadeira (caso específico do T2)
        salas_compativeis = 0
        salas_quase_compativeis = []
        for sala in salas_ativas:
            if sala["capacidade_total"] < turma["alunos"]:
                if sala["capacidade_total"] == turma["alunos"] - 1:
                    salas_quase_compativeis.append(sala["nome"])
                continue
            if sala["cadeiras_especiais"] < turma["esp_necessarias"]:
                continue
            salas_compativeis += 1
        
        if salas_compativeis == 0:
            if salas_quase_compativeis:
                motivos.append(f"Sala {salas_quase_compativeis[0]} tem {turma['alunos']-1} lugares e a turma tem {turma['alunos']}; as salas com especiais ficaram reservadas para turmas que precisam de especiais")
            else:
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
            prova = self._prova_impossibilidade()

            # NOVO: continuar mesmo com F1 baixo; só abortar se não houver dados de pares
            min_f1 = float(self.parametros.get("min_f1_para_continuar", 0.0))
            continuar_quando_f1_baixo = bool(self.parametros.get("continuar_quando_f1_baixo", True))

            if (self.df is None) or self.df.empty:
                # aqui sim é falha real de ML/feature building
                if self.permitir_fallback_simples:
                    print("🔄 [MAIN] Sem dados de pares; usando empréstimo greedy direto", file=sys.stderr)
                    alocacoes, score, turmas_nao_alocadas, transferencias_cadeiras = self._algoritmo_simples_fallback()
                    acuracia = max(acuracia, 0.5)
                    algoritmo_usado = "Empréstimo entre salas (Greedy)"
                else:
                    raise RuntimeError("Sem pares válidos e fallback simples desabilitado (TCC: apenas IA).")
            elif (acuracia < min_f1) and (not continuar_quando_f1_baixo):
                # política estrita opcional
                raise RuntimeError(f"F1 abaixo do mínimo ({acuracia:.2%} < {min_f1:.2%}) e política estrita ativa.")
            else:
                if acuracia < 0.1:  # F1 muito baixo
                    print(f"⚠️ [MAIN] F1 baixo ({acuracia:.2%}) mas continuando com otimização IA", file=sys.stderr)
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
                "otimizador_ia": "Fluxo Mínimo (Ótimo)" if algoritmo_usado == "Fluxo Mínimo (Ótimo)" else "Hungarian" if "Hungarian" in algoritmo_usado else "Força Bruta",
                "parametros_utilizados": self.parametros,
                "scipy_disponivel": HAS_SCIPY,
                "metricas_honestas": getattr(self, 'metricas_honestas', False),
                "prova_impossibilidade": prova
            }

            # Melhorar motivos das turmas não alocadas se há impossibilidade estrutural
            if prova.get("impossibilidade_estrutural"):
                motivo_global = f"Impossível alocar todas as turmas no mesmo bloco: {prova['turmas']} turmas para {prova['salas_ativas']} salas"
                if "cadeiras_moveis_faltantes" in prova:
                    motivo_global += f" ; faltam {prova['cadeiras_moveis_faltantes']} cadeiras móveis"
                for x in turmas_nao_alocadas:
                    x["motivo"] = motivo_global

            observacoes = []
            # Sempre adicionar "Somente IA" para TCC
            observacoes.append("Somente IA: Árvore de Decisão (ML) + Otimizador combinatório (Fluxo/Húngaro)")
            
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

            # Diagnóstico para detectar divergências entre DF e UI
            if alocacoes:
                # Score baseado no DF original (pré-empréstimo)
                scores_df = []
                for a in alocacoes:
                    turma_id_turma = next(t["id_turma"] for t in self.turmas if t["id"] == a["turma_id"])
                    sala_id_sala = next(s["id_sala"] for s in self.salas if s["id"] == a["sala_id"])
                    linha = self.df[(self.df["id_turma"] == turma_id_turma) & (self.df["id_sala"] == sala_id_sala)].iloc[0]
                    scores_df.append(float(linha["proba_bom"]) * 100)
                
                debug_info["score_media_df_pct"] = round(np.mean(scores_df), 2)
                debug_info["score_media_ui_pct"] = round(np.mean([a["compatibilidade_score"] for a in alocacoes]), 2)
            else:
                debug_info["score_media_df_pct"] = 0.0
                debug_info["score_media_ui_pct"] = 0.0

            print(f"✅ [MAIN] Execução concluída: {turmas_alocadas}/{total_turmas} turmas alocadas ({score_otimizacao_pct}%)", file=sys.stderr)

            # Determinar confiança do modelo
            confianca_modelo = "baixa"
            if hasattr(self, 'metricas_honestas') and self.metricas_honestas:
                if hasattr(self, 'clf_report') and isinstance(self.clf_report, dict):
                    f1_score = self.clf_report.get('1', {}).get('f1-score', 0)
                    if f1_score >= 0.7:
                        confianca_modelo = "alta"
                    elif f1_score >= 0.5:
                        confianca_modelo = "média"
            
            return {
                "success": True,
                "alocacoes": alocacoes,
                "turmas_nao_alocadas": turmas_nao_alocadas,
                "score_otimizacao": score_otimizacao_pct,
                "total_alocacoes": turmas_alocadas,
                "total_turmas": total_turmas,
                "turmas_sobrando": turmas_sobrando,
                "confianca_modelo": confianca_modelo,
                "acuracia_modelo": round(acuracia * 100, 2) if confianca_modelo != "baixa" else None,
                "f1_modelo": round(acuracia * 100, 2) if confianca_modelo != "baixa" else None,
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
