#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algoritmo Inteligente de Alocação com Machine Learning
Integrado com backend Node.js para sistema de gerenciamento de alocação
Autor: Sistema de Alocação Inteligente
"""

import json
import sys
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
        """Normaliza dados das salas para formato esperado"""
        normalized = []
        for sala in salas:
            normalized.append({
                "id": sala.get("id"),
                "id_sala": sala.get("id_sala", sala.get("id")),
                "nome": sala.get("nome", ""),
                "capacidade_total": int(sala.get("capacidade_total", 0)),
                "localizacao": sala.get("localizacao", ""),
                "status": sala.get("status", "ATIVA").upper(),
                "cadeiras_moveis": bool(sala.get("cadeiras_moveis", True)),
                "cadeiras_especiais": int(sala.get("cadeiras_especiais", 0))
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
                sala_movel = 1 if s["cadeiras_moveis"] else 0

                # especiais
                esp_need = int(t.get("esp_necessarias", 0))
                esp_have = int(s.get("cadeiras_especiais", 0))
                esp_deficit = max(0, esp_need - esp_have)
                esp_sobra = max(0, esp_have - esp_need)
                atende_especial = 1 if esp_deficit == 0 else 0

                # Score de ocupação (alvo ~85%)
                ocupacao = alunos / cap if cap > 0 else 0.0
                score_ocupacao = 1 - min(abs(ocupacao - 0.85) / 0.85, 1) if cap > 0 else 0.0

                # RÓTULO HEURÍSTICO (apenas se atende cadeiras especiais)
                within = (0.70 <= ocupacao <= 0.95)
                deficit_pequeno = (deficit <= 3) and (deficit > 0)
                match_bom = 1 if (atende_especial and ((cabe_direto and within) or (not cabe_direto and sala_movel and deficit_pequeno))) else 0

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
        # Construir features
        self.df = self.build_pair_features()
        
        if self.df.empty:
            raise Exception("Nenhum par turma-sala válido encontrado")
        
        feature_cols = [
            "alunos", "capacidade_total", "deficit", "sobra_local",
            "sala_movel", "ocupacao", "score_ocupacao",
            "esp_necessarias", "esp_disponiveis", "esp_deficit", "esp_sobra", "atende_especial"
        ]
        
        X = self.df[feature_cols].values
        y = self.df["label_match_bom"].values
        
        counts = Counter(y)
        strat = y if min(counts.values()) >= 2 else None
        
        if len(X) < 4:  # Poucos dados para split
            X_train, X_test, y_train, y_test = X, X, y, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=strat
            )
        
        self.clf = DecisionTreeClassifier(
            max_depth=4,
            min_samples_split=2,
            random_state=42
        )
        self.clf.fit(X_train, y_train)
        
        # Calcular probabilidades
        try:
            proba_result = self.clf.predict_proba(X)
            if proba_result.shape[1] > 1:
                self.df["proba_bom"] = proba_result[:, 1]
            else:
                # Caso com apenas uma classe
                self.df["proba_bom"] = proba_result[:, 0]
        except Exception:
            # Fallback: usar predições binárias
            self.df["proba_bom"] = self.clf.predict(X).astype(float)
        
        return self.clf.score(X_test, y_test)

    def par_viavel(self, row):
        """Verifica se um par turma-sala é viável"""
        if row["esp_deficit"] > 0:
            return False
        return (row["deficit"] == 0) or (row["sala_movel"] == 1 and row["deficit"] <= 3)

    def otimizar_alocacoes(self):
        """Executa otimização global das alocações"""
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
        
        alocacoes_otimas = []
        if best_assign is not None:
            for i in range(n):
                j = best_assign[i]
                linha = self.df[(self.df["id_turma"] == turma_ids[i]) & (self.df["id_sala"] == sala_ids[j])].iloc[0]
                
                # Encontrar IDs reais das entidades
                turma_real = next(t for t in self.turmas if t["id_turma"] == turma_ids[i])
                sala_real = next(s for s in self.salas if s["id_sala"] == sala_ids[j])
                
                alocacoes_otimas.append({
                    "sala_id": sala_real["id"],
                    "turma_id": turma_real["id"],
                    "compatibilidade_score": round(float(linha["proba_bom"]) * 100, 2),
                    "observacoes": f"Ocupacao: {linha['ocupacao']:.1%}, Especiais: {linha['esp_necessarias']}/{linha['esp_disponiveis']}"
                })
        
        return alocacoes_otimas, best_score

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
                        avisos.append(f"Turma '{t['nome']}' em '{s['nome']}': ocupação de {ocupacao:.1%} (acima de 100%)")
                    elif ocupacao < 0.5:
                        avisos.append(f"Turma '{t['nome']}' em '{s['nome']}': baixa ocupação de {ocupacao:.1%} (desperdício de espaço)")
            
            if turma_opcoes == 0:
                problemas.append({
                    "turma": t["nome"],
                    "tipo": "sem_opcoes",
                    "detalhes": turma_problemas,
                    "resumo": f"Turma '{t['nome']}' ({t['alunos']} alunos, {t['esp_necessarias']} especiais) não tem nenhuma sala compatível"
                })
            elif turma_opcoes == 1:
                avisos.append(f"Turma '{t['nome']}' tem apenas 1 opção de sala - flexibilidade limitada")
        
        # Verificar conflitos de salas
        if len(self.salas) < len(self.turmas):
            problemas.append({
                "tipo": "salas_insuficientes", 
                "resumo": f"Há {len(self.turmas)} turmas para apenas {len(self.salas)} salas - conflitos inevitáveis"
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
            
            # Otimizar alocações
            alocacoes, score = self.otimizar_alocacoes()
            
            return {
                "success": True,
                "alocacoes": alocacoes,
                "score_otimizacao": round(score * 100 / len(self.turmas), 2) if alocacoes else 0,
                "total_alocacoes": len(alocacoes),
                "acuracia_modelo": round(acuracia * 100, 2),
                "analise_detalhada": analise,
                "debug_info": {
                    "total_pares_avaliados": len(self.df),
                    "pares_viaveis": len(self.df[self.df.apply(self.par_viavel, axis=1)]),
                    "salas_ativas": len([s for s in self.salas if s["status"].upper() == "ATIVA"])
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "alocacoes": [],
                "score_otimizacao": 0
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
        print(f"⚙️ [PYTHON] Parâmetros: {parametros}", file=sys.stderr)
        
        # Executar algoritmo
        print("🤖 [PYTHON] Criando instância do algoritmo...", file=sys.stderr)
        algoritmo = AlocacaoInteligenteMLA(dados['salas'], dados['turmas'], parametros)
        
        print("⚡ [PYTHON] Executando algoritmo...", file=sys.stderr)
        resultado = algoritmo.executar()
        
        print(f"✅ [PYTHON] Algoritmo executado! Success: {resultado.get('success')}", file=sys.stderr)
        print(f"📈 [PYTHON] Resultado: {len(resultado.get('alocacoes', []))} alocações, score: {resultado.get('score_otimizacao')}", file=sys.stderr)
        
        print(json.dumps(resultado, ensure_ascii=False))
        
        if not resultado['success']:
            print(f"❌ [PYTHON] Falha no algoritmo: {resultado.get('error')}", file=sys.stderr)
            sys.exit(1)
            
    except Exception as e:
        print(f"💥 [PYTHON] Exceção capturada: {str(e)}", file=sys.stderr)
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
