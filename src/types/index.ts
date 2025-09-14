import { z } from 'zod';

// Schemas de validação com Zod
export const SalaSchema = z.object({
  id_sala: z.string().min(1, 'ID da sala é obrigatório'),
  nome: z.string().min(1, 'Nome é obrigatório'),
  capacidade_total: z.number().min(1, 'Capacidade deve ser maior que 0'),
  localizacao: z.string().min(1, 'Localização é obrigatória'),
  status: z.enum(['ATIVA', 'INATIVA', 'MANUTENCAO']).default('ATIVA'),
  cadeiras_moveis: z.number().min(0).default(0),
  cadeiras_especiais: z.number().min(0).default(0)
});

export const TurmaSchema = z.object({
  id_turma: z.string().min(1, 'ID da turma é obrigatório'),
  nome: z.string().min(1, 'Nome é obrigatório'),
  alunos: z.number().min(1, 'Número de alunos deve ser maior que 0'),
  esp_necessarias: z.number().min(0).default(0)
});

export const ProjetoAlocacaoSchema = z.object({
  id_projeto: z.string().min(1, 'ID do projeto é obrigatório'),
  nome: z.string().min(1, 'Nome é obrigatório'),
  descricao: z.string().min(1, 'Descrição é obrigatória'),
  status: z.enum(['CONFIGURACAO', 'PRONTO', 'PROCESSANDO', 'ALOCADO', 'CONCLUIDO']).default('CONFIGURACAO')
});

export const ResultadoAlocacaoSchema = z.object({
  projeto_id: z.string().min(1, 'ID do projeto é obrigatório'),
  score_otimizacao: z.number().min(0).max(100),
  priorizar_capacidade: z.boolean().default(true),
  priorizar_especiais: z.boolean().default(true),
  priorizar_proximidade: z.boolean().default(true),
  alocacoes: z.array(z.object({
    sala_id: z.string(),
    turma_id: z.string(),
    compatibilidade_score: z.number().min(0).max(100),
    observacoes: z.string().optional()
  }))
});

// Tipos TypeScript derivados dos schemas
export type CreateSalaData = z.infer<typeof SalaSchema>;
export type CreateTurmaData = z.infer<typeof TurmaSchema>;
export type CreateProjetoAlocacaoData = z.infer<typeof ProjetoAlocacaoSchema>;
export type CreateResultadoAlocacaoData = z.infer<typeof ResultadoAlocacaoSchema>;

// Tipos para update (todos os campos opcionais)
export type UpdateSalaData = Partial<CreateSalaData>;
export type UpdateTurmaData = Partial<CreateTurmaData>;
export type UpdateProjetoAlocacaoData = Partial<CreateProjetoAlocacaoData>;

// Tipos de resposta da API
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface PaginatedResponse<T> extends ApiResponse<T[]> {
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
  };
}
