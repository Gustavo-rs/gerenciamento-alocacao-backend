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
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


class AlocacaoInteligenteMLA:
    """Classe principal para algoritmo de alocação com Machine Learning"""
    
    def __init__(self, salas, turmas, parametros):
        self.salas = self._normalize_salas(salas)
        self.turmas = self._normalize_turmas(turmas)
        self.parametros = parametros
        self.clf = None
        self.df = None
        
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
                # mantém a quantidade e um flag derivado
                "cadeiras_moveis_qtd": moveis_qtd,
                "cadeiras_moveis": bool(moveis_qtd > 0),
                "cadeiras_especiais": int(sala.get("cadeiras_especiais", 0) or 0),
            })
        return normalized
        
    def _normalize_turmas(self, turmas):
        """Normaliza dados das turmas para formato esperado"""
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
        rows = []
        for t in self.turmas:
            for s in self.salas:
                # Verificar se sala está ativa
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

                # Score de ocupação (alvo ~85%, mas 100% é aceitável)
                ocupacao = alunos / cap if cap > 0 else 0.0
                if ocupacao <= 1.0:  # Até 100% é bom
                    score_ocupacao = 1 - min(abs(ocupacao - 0.85) / 0.85, 1) if cap > 0 else 0.0
                else:  # Acima de 100% é problemático
                    score_ocupacao = max(0, 0.5 - (ocupacao - 1.0))  # Penalizar superlotação

                # RÓTULO HEURÍSTICO MELHORADO
                within = (0.70 <= ocupacao <= 1.0)  # Permitir até 100% ocupação
                deficit_pequeno = (deficit <= 3) and (deficit > 0)
                
                # Preferir salas com melhor aproveitamento
                aproveitamento_bom = ocupacao >= 0.75  # Pelo menos 75% de ocupação
                
                # Match é bom se:
                # 1. Atende especiais E ocupação boa E (cabe direto OU sala móvel com déficit pequeno)
                match_bom = 1 if (atende_especial and aproveitamento_bom and ((cabe_direto) or (not cabe_direto and sala_movel and deficit_pequeno))) else 0

                rows.append({
                    "id_turma": t["id_turma"],
                    "id_sala": s["id_sala"],
                    "alunos": alunos,
                    "capacidade_total": cap,
                    "deficit": deficit,
                    "sobra_local": sobra,
                    "sala_movel": sala_movel,
                    "ocupacao": ocupacao,
                    "score_ocupacao": score_ocupacao,
                    # especiais
                    "esp_necessarias": esp_need,
                    "esp_disponiveis": esp_have,
                    "esp_deficit": esp_deficit,
                    "esp_sobra": esp_sobra,
                    "atende_especial": atende_especial,
                    "label_match_bom": match_bom
                })
        return pd.DataFrame(rows)

    def treinar_modelo(self):
        """Treina o modelo de Machine Learning"""
        try:
            # Construir features
            self.df = self.build_pair_features()
            
            if self.df.empty:
                print("❌ [PYTHON] Nenhum par turma-sala valido encontrado", file=sys.stderr)
                raise Exception("Nenhum par turma-sala válido encontrado")
            
            feature_cols = [
                "alunos", "capacidade_total", "deficit", "sobra_local",
                "sala_movel", "ocupacao", "score_ocupacao",
                "esp_necessarias", "esp_disponiveis", "esp_deficit", "esp_sobra", "atende_especial"
            ]
            
            X = self.df[feature_cols].values
            y = self.df["label_match_bom"].values
            
            print(f"📊 [PYTHON] Dataset: {len(X)} pares, classes: {set(y)}", file=sys.stderr)
            
            counts = Counter(y)
            
            # Para datasets muito pequenos, usar todo o dataset para treinamento e teste
            if len(X) < 4 or len(set(y)) < 2:
                X_train, X_test, y_train, y_test = X, X, y, y
                print(f"📊 [PYTHON] Dataset pequeno: usando {len(X)} amostras para treino/teste", file=sys.stderr)
            else:
                strat = y if min(counts.values()) >= 2 else None
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.25, random_state=42, stratify=strat
                    )
                except ValueError as e:
                    print(f"⚠️ [PYTHON] Erro no split dos dados: {e}. Usando dataset completo.", file=sys.stderr)
                    X_train, X_test, y_train, y_test = X, X, y, y
            
            self.clf = DecisionTreeClassifier(
                max_depth=4,
                min_samples_split=2,
                random_state=42
            )
            self.clf.fit(X_train, y_train)
            
            try:
                proba_result = self.clf.predict_proba(X)
                if proba_result.shape[1] > 1:
                    proba_ml = proba_result[:, 1]
                    self.df["proba_ml"] = proba_ml  # <-- guarda o ML puro
                    score_ocupacao = self.df["score_ocupacao"].values

                    # combinado (0..1) e com clamp
                    proba_comb = (0.4 * proba_ml) + (0.6 * score_ocupacao)
                    self.df["proba_bom"] = np.clip(proba_comb, 0.0, 1.0)
                    print(f"✅ [PYTHON] Probabilidades combinadas ML(40%)+ocupacao(60%) calculadas para {len(X)} pares", file=sys.stderr)
                    
                    # --- Regra de "match perfeito" (força 100%) ---
                    # perfeição = ocupa 100% + sem déficit de especiais + sem déficit de capacidade
                    mask_perfeito = (
                        np.isclose(self.df["ocupacao"].values, 1.0) &
                        (self.df["esp_deficit"].values == 0) &
                        (self.df["deficit"].values == 0)
                    )
                    
                    # Ajusta o score combinado para 1.0 nos casos perfeitos
                    self.df.loc[mask_perfeito, "proba_bom"] = 1.0
                    
                    # Marcar flag para aparecer nas observações
                    self.df["match_perfeito"] = False
                    self.df.loc[mask_perfeito, "match_perfeito"] = True
                    # --- fim do bloco ---
                else:
                    print(f"⚠️ [PYTHON] Apenas uma classe detectada, usando score de ocupacao", file=sys.stderr)
                    self.df["proba_ml"] = 0.0
                    self.df["proba_bom"] = self.df["score_ocupacao"]
                    
                    # --- Regra de "match perfeito" (força 100%) ---
                    mask_perfeito = (
                        np.isclose(self.df["ocupacao"].values, 1.0) &
                        (self.df["esp_deficit"].values == 0) &
                        (self.df["deficit"].values == 0)
                    )
                    
                    # Ajusta o score combinado para 1.0 nos casos perfeitos
                    self.df.loc[mask_perfeito, "proba_bom"] = 1.0
                    
                    # Marcar flag para aparecer nas observações
                    self.df["match_perfeito"] = False
                    self.df.loc[mask_perfeito, "match_perfeito"] = True
                    # --- fim do bloco ---
                    
            except Exception as e:
                print(f"⚠️ [PYTHON] Erro ao calcular probabilidades: {e}", file=sys.stderr)
                self.df["proba_ml"] = 0.0
                self.df["proba_bom"] = self.df["score_ocupacao"]
                
                # --- Regra de "match perfeito" (força 100%) ---
                mask_perfeito = (
                    np.isclose(self.df["ocupacao"].values, 1.0) &
                    (self.df["esp_deficit"].values == 0) &
                    (self.df["deficit"].values == 0)
                )
                
                # Ajusta o score combinado para 1.0 nos casos perfeitos
                self.df.loc[mask_perfeito, "proba_bom"] = 1.0
                
                # Marcar flag para aparecer nas observações
                self.df["match_perfeito"] = False
                self.df.loc[mask_perfeito, "match_perfeito"] = True
                # --- fim do bloco ---
            
            accuracy = self.clf.score(X_test, y_test)
            print(f"🎯 [PYTHON] Modelo treinado com acuracia: {accuracy:.2%}", file=sys.stderr)
            return accuracy
            
        except Exception as e:
            print(f"❌ [PYTHON] Erro critico no treinamento: {e}", file=sys.stderr)
            # Se o ML falha completamente, retorna acuracia 0 e deixa o fallback lidar
            return 0.0

    def par_viavel(self, row):
        """Verifica se um par turma-sala e viavel"""
        if row["esp_deficit"] > 0:
            return False
        # Nao permitir ocupacao > 100% (mesmo com moveis)
        if row["ocupacao"] > 1.0:
            return False
        return (row["deficit"] == 0) or (row["sala_movel"] == 1 and row["deficit"] <= 3)

    def _algoritmo_simples_fallback(self):
        """Algoritmo simples de alocação quando ML falha"""
        print("🔄 [PYTHON] Usando algoritmo simples de fallback", file=sys.stderr)
        
        alocacoes_simples = []
        turmas_alocadas = set()
        salas_usadas = set()
        turmas_nao_alocadas = []
        
        # Ordenar turmas por número de alunos (maiores primeiro)
        turmas_ordenadas = sorted(self.turmas, key=lambda t: t["alunos"], reverse=True)
        
        # Ordenar salas por capacidade (maiores primeiro)
        salas_ativas = [s for s in self.salas if s["status"].upper() == "ATIVA"]
        salas_ordenadas = sorted(salas_ativas, key=lambda s: s["capacidade_total"], reverse=True)
        
        for turma in turmas_ordenadas:
            melhor_sala = None
            melhor_score = -1
            
            for sala in salas_ordenadas:
                if sala["id"] in salas_usadas:
                    continue
                    
                # Verificar se a sala pode acomodar a turma
                if sala["capacidade_total"] < turma["alunos"]:
                    continue
                    
                # Verificar cadeiras especiais
                if sala["cadeiras_especiais"] < turma["esp_necessarias"]:
                    continue
                
                # Calcular score simples (ocupação próxima a 85%)
                ocupacao = turma["alunos"] / sala["capacidade_total"]
                score = max(0, 1 - abs(ocupacao - 0.85))
                if ocupacao == 1.0 and sala["cadeiras_especiais"] == turma["esp_necessarias"]:
                    score = 1.0

                                
                # Bônus por atender exatamente as cadeiras especiais
                if sala["cadeiras_especiais"] >= turma["esp_necessarias"]:
                    if sala["cadeiras_especiais"] == turma["esp_necessarias"]:
                        score += 0.1  # Bônus por match exato
                    score = min(1.0, score)  # Garantir que não passe de 1.0
                
                if score > melhor_score:
                    melhor_score = score
                    melhor_sala = sala
            
            if melhor_sala:
                ocupacao = turma["alunos"] / melhor_sala["capacidade_total"]
                
                # Explicar como foi calculado o score
                score_base = max(0, 1 - abs(ocupacao - 0.85))
                obs_detalhes = f"Ocupacao: {ocupacao:.1%} (score base: {score_base:.2f})"
                
                if melhor_sala["cadeiras_especiais"] == turma["esp_necessarias"]:
                    obs_detalhes += f", Match exato especiais (+10%)"
                
                obs_detalhes += f", Especiais: {turma['esp_necessarias']}/{melhor_sala['cadeiras_especiais']} (Algoritmo Simples)"
                
                alocacoes_simples.append({
                    "sala_id": melhor_sala["id"],
                    "turma_id": turma["id"],
                    "compatibilidade_score": round(melhor_score * 100, 2),
                    "observacoes": obs_detalhes
                })
                turmas_alocadas.add(turma["id"])
                salas_usadas.add(melhor_sala["id"])
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
        
        # Calcular score com denominador correto (máximo de matches possíveis)
        salas_ativas = [s for s in self.salas if s["status"].upper() == "ATIVA"]
        den = max(1, min(len(self.turmas), len(salas_ativas)))
        score_total = len(alocacoes_simples) / den if den > 0 else 0
        return alocacoes_simples, score_total, turmas_nao_alocadas

    def otimizar_alocacoes(self):
        """Executa otimização global das alocações"""
        try:
            if self.df is None:
                self.treinar_modelo()
            
            turma_ids = [t["id_turma"] for t in self.turmas]
            sala_ids = [s["id_sala"] for s in self.salas]
            idx_turma = {tid: i for i, tid in enumerate(turma_ids)}
            idx_sala = {sid: i for i, sid in enumerate(sala_ids)}
            
            # Matriz de utilidade
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
            
            # Força bruta para otimização (funciona bem para pequenos datasets)
            for salas_perm in permutations(range(n_s), n):
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
        except Exception as e:
            print(f"⚠️ [PYTHON] Erro na otimizacao ML: {e}. Usando algoritmo simples.", file=sys.stderr)
            return self._algoritmo_simples_fallback()
        
        alocacoes_otimas = []
        turmas_alocadas = set()
        turmas_nao_alocadas = []
        
        if best_assign is not None:
            for i in range(n):
                j = best_assign[i]
                linha = self.df[(self.df["id_turma"] == turma_ids[i]) & (self.df["id_sala"] == sala_ids[j])].iloc[0]
                
                # Encontrar IDs reais das entidades
                turma_real = next(t for t in self.turmas if t["id_turma"] == turma_ids[i])
                sala_real = next(s for s in self.salas if s["id_sala"] == sala_ids[j])
                
                turmas_alocadas.add(turma_real["id"])
                
                # Calcular detalhes para observação (com dados corretos)
                ocupacao = linha['ocupacao']
                score_base = linha['score_ocupacao']
                score_ml = float(linha.get('proba_ml', 0.0))
                score_final = float(linha['proba_bom'])  # combinado
                
                obs_detalhes = (
                    f"Ocupacao: {ocupacao:.1%} (score ocupacao: {score_base:.2f}), "
                    f"Score ML: {score_ml:.2f}, "
                    f"Score combinado: {score_final:.2f}, "
                    f"Especiais: {linha['esp_necessarias']}/{linha['esp_disponiveis']} (ML)"
                )
                
                # Indicar se foi match perfeito
                if bool(linha.get("match_perfeito", False)):
                    obs_detalhes += ", Match perfeito (forcado a 100%)"
                
                alocacoes_otimas.append({
                    "sala_id": sala_real["id"],
                    "turma_id": turma_real["id"],
                    "compatibilidade_score": round(score_final * 100, 2),
                    "observacoes": obs_detalhes
                })
            
            # Identificar turmas que não foram alocadas
            for turma in self.turmas:
                if turma["id"] not in turmas_alocadas:
                    turmas_nao_alocadas.append({
                        "id": turma["id"],
                        "nome": turma["nome"],
                        "alunos": turma["alunos"],
                        "esp_necessarias": turma["esp_necessarias"],
                        "motivo": self._analisar_motivo_nao_alocacao(turma)
                    })
            
            print(f"✅ [PYTHON] ML encontrou {len(alocacoes_otimas)} alocacoes com score {best_score:.2f}", file=sys.stderr)
            return alocacoes_otimas, best_score, turmas_nao_alocadas
        else:
            # ML não conseguiu encontrar solução, usar algoritmo simples
            print("⚠️ [PYTHON] ML nao encontrou solucoes viaveis. Usando algoritmo simples.", file=sys.stderr)
            return self._algoritmo_simples_fallback()

    def _analisar_motivo_nao_alocacao(self, turma):
        """Analisa por que uma turma nao foi alocada"""
        motivos = []
        
        # Verificar se ha salas disponiveis
        salas_ativas = [s for s in self.salas if s["status"].upper() == "ATIVA"]
        if len(salas_ativas) == 0:
            return "Nenhuma sala ativa disponivel"
        
        # Verificar se ha mais turmas que salas
        if len(self.turmas) > len(salas_ativas):
            motivos.append(f"Mais turmas ({len(self.turmas)}) que salas ({len(salas_ativas)})")
        
        # Verificar compatibilidade com salas
        salas_compativeis = 0
        for sala in salas_ativas:
            # Verificar capacidade
            if sala["capacidade_total"] < turma["alunos"]:
                continue
            # Verificar cadeiras especiais
            if sala["cadeiras_especiais"] < turma["esp_necessarias"]:
                continue
            salas_compativeis += 1
        
        if salas_compativeis == 0:
            motivos.append("Nenhuma sala compativel (capacidade ou cadeiras especiais)")
        elif salas_compativeis < len(self.turmas):
            motivos.append("Poucas salas compativeis para todas as turmas")
        
        return "; ".join(motivos) if motivos else "Salas insuficientes para otimizacao"

    def _analisar_problemas(self):
        """Analisa problemas detalhados da alocação"""
        problemas = []
        avisos = []
        
        for t in self.turmas:
            turma_problemas = []
            turma_opcoes = 0
            
            for s in self.salas:
                if s["status"].upper() != "ATIVA":
                    continue
                    
                # Verificar problemas específicos
                if s["capacidade_total"] < t["alunos"]:
                    deficit = t["alunos"] - s["capacidade_total"]
                    turma_problemas.append(f"Sala '{s['nome']}': capacidade insuficiente ({s['capacidade_total']} < {t['alunos']}, faltam {deficit} lugares)")
                
                if s["cadeiras_especiais"] < t["esp_necessarias"]:
                    deficit_esp = t["esp_necessarias"] - s["cadeiras_especiais"]
                    turma_problemas.append(f"Sala '{s['nome']}': cadeiras especiais insuficientes ({s['cadeiras_especiais']} < {t['esp_necessarias']}, faltam {deficit_esp})")
                
                # Verificar se é viável
                ocupacao = t["alunos"] / s["capacidade_total"] if s["capacidade_total"] > 0 else 0
                if (s["capacidade_total"] >= t["alunos"] and 
                    s["cadeiras_especiais"] >= t["esp_necessarias"]):
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
        
        # Verificar conflitos de salas
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
        try:
            # Validações básicas
            if not self.salas:
                raise Exception("Nenhuma sala fornecida")
            if not self.turmas:
                raise Exception("Nenhuma turma fornecida")
            
            # Treinar modelo
            acuracia = self.treinar_modelo()
            
            # Análise detalhada dos problemas
            analise = self._analisar_problemas()
            
            # Se o treinamento falhou completamente, usar algoritmo simples direto
            if acuracia == 0.0 or self.df is None or self.df.empty:
                print("🔄 [PYTHON] ML falhou completamente, usando algoritmo simples direto", file=sys.stderr)
                alocacoes, score, turmas_nao_alocadas = self._algoritmo_simples_fallback()
                acuracia = 0.5  # Acurácia estimada para algoritmo simples
            else:
                # Otimizar alocações com ML
                alocacoes, score, turmas_nao_alocadas = self.otimizar_alocacoes()
            
            # Calcular estatísticas com denominador correto
            total_turmas = len(self.turmas)
            total_salas_ativas = len([s for s in self.salas if s["status"].upper() == "ATIVA"])
            turmas_alocadas = len(alocacoes)
            turmas_sobrando = len(turmas_nao_alocadas)
            
            # Número máximo de matches possíveis
            den = max(1, min(total_turmas, total_salas_ativas))
            score_otimizacao_pct = round((score / den) * 100, 2) if alocacoes else 0
            
            return {
                "success": True,
                "alocacoes": alocacoes,
                "turmas_nao_alocadas": turmas_nao_alocadas,
                "score_otimizacao": score_otimizacao_pct,
                "total_alocacoes": turmas_alocadas,
                "total_turmas": total_turmas,
                "turmas_sobrando": turmas_sobrando,
                "acuracia_modelo": round(acuracia * 100, 2),
                "analise_detalhada": analise,
                "debug_info": {
                    "total_pares_avaliados": len(self.df) if hasattr(self, 'df') and self.df is not None else 0,
                    "pares_viaveis": len(self.df[self.df.apply(self.par_viavel, axis=1)]) if hasattr(self, 'df') and self.df is not None else 0,
                    "salas_ativas": total_salas_ativas,
                    "turmas_vs_salas": f"{total_turmas} turmas para {total_salas_ativas} salas",
                    "max_matches_possiveis": den
                }
            }
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"💥 [PYTHON] Erro completo: {error_details}", file=sys.stderr)
            
            # Diagnóstico mais detalhado
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
    parser.add_argument('--parametros', required=True, help='JSON com parâmetros de otimização')
    
    args = parser.parse_args()
    print(f"📝 [PYTHON] Argumentos recebidos: dados={len(args.dados)} chars, parametros={len(args.parametros)} chars", file=sys.stderr)
    
    try:
        # Carregar dados
        print("🔍 [PYTHON] Parseando dados JSON...", file=sys.stderr)
        dados = json.loads(args.dados)
        parametros = json.loads(args.parametros)
        
        print(f"📊 [PYTHON] Dados parseados: {len(dados.get('salas', []))} salas, {len(dados.get('turmas', []))} turmas", file=sys.stderr)
        print(f"⚙️ [PYTHON] Parametros: {parametros}", file=sys.stderr)
        
        # Executar algoritmo
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
