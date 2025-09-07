import { Router } from 'express';
import { ResultadoAlocacaoSchema, ApiResponse } from '../types';
import { z } from 'zod';
import { pythonService } from '../services/pythonService';

const router = Router();

// GET /api/resultados/test-python - Testar integração Python
router.get('/test-python', async (req, res) => {
  try {
    // Verificar se Python está disponível
    const pythonDisponivel = await pythonService.verificarPython();
    
    if (!pythonDisponivel) {
      return res.status(500).json({
        success: false,
        error: 'Python não está instalado ou disponível no PATH'
      });
    }

    // Testar script
    const scriptFuncionando = await pythonService.testarScript();
    
    if (!scriptFuncionando) {
      return res.status(500).json({
        success: false,
        error: 'Script Python não está funcionando corretamente'
      });
    }

    res.json({
      success: true,
      message: 'Integração Python funcionando perfeitamente!',
      python_disponivel: pythonDisponivel,
      script_funcionando: scriptFuncionando
    });

  } catch (error) {
    console.error('Erro no teste Python:', error);
    res.status(500).json({
      success: false,
      error: 'Erro ao testar integração Python'
    });
  }
});

// GET /api/resultados - Listar todos os resultados
router.get('/', async (req, res) => {
  try {
    const resultados = await req.prisma.resultadoAlocacao.findMany({
      include: {
        projeto: true,
        alocacoes: {
          include: {
            sala: true,
            turma: true
          }
        }
      },
      orderBy: { created_at: 'desc' }
    });

    const response: ApiResponse = {
      success: true,
      data: resultados
    };

    res.json(response);
  } catch (error) {
    console.error('Erro ao buscar resultados:', error);
    res.status(500).json({
      success: false,
      error: 'Erro interno do servidor'
    });
  }
});

// GET /api/resultados/:id - Buscar resultado por ID
router.get('/:id', async (req, res) => {
  try {
    const { id } = req.params;

    const resultado = await req.prisma.resultadoAlocacao.findUnique({
      where: { id },
      include: {
        projeto: true,
        alocacoes: {
          include: {
            sala: true,
            turma: true
          }
        }
      }
    });

    if (!resultado) {
      return res.status(404).json({
        success: false,
        error: 'Resultado não encontrado'
      });
    }

    const response: ApiResponse = {
      success: true,
      data: resultado
    };

    res.json(response);
  } catch (error) {
    console.error('Erro ao buscar resultado:', error);
    res.status(500).json({
      success: false,
      error: 'Erro interno do servidor'
    });
  }
});

// GET /api/resultados/projeto/:projetoId - Buscar resultados por projeto
router.get('/projeto/:projetoId', async (req, res) => {
  try {
    const { projetoId } = req.params;

    const resultados = await req.prisma.resultadoAlocacao.findMany({
      where: { projeto_id: projetoId },
      include: {
        projeto: true,
        alocacoes: {
          include: {
            sala: true,
            turma: true
          }
        }
      },
      orderBy: { created_at: 'desc' }
    });

    const response: ApiResponse = {
      success: true,
      data: resultados
    };

    res.json(response);
  } catch (error) {
    console.error('Erro ao buscar resultados do projeto:', error);
    res.status(500).json({
      success: false,
      error: 'Erro interno do servidor'
    });
  }
});

// POST /api/resultados - Criar novo resultado de alocação
router.post('/', async (req, res) => {
  try {
    const validatedData = ResultadoAlocacaoSchema.parse(req.body);

    // Verificar se o projeto existe
    const projeto = await req.prisma.projetoAlocacao.findUnique({
      where: { id: validatedData.projeto_id }
    });

    if (!projeto) {
      return res.status(404).json({
        success: false,
        error: 'Projeto não encontrado'
      });
    }

    // Usar transação para criar resultado e alocações
    const resultado = await req.prisma.$transaction(async (prisma) => {
      // Criar resultado
      const novoResultado = await prisma.resultadoAlocacao.create({
        data: {
          projeto_id: validatedData.projeto_id,
          score_otimizacao: validatedData.score_otimizacao,
          priorizar_capacidade: validatedData.priorizar_capacidade,
          priorizar_especiais: validatedData.priorizar_especiais,
          priorizar_proximidade: validatedData.priorizar_proximidade
        }
      });

      // Criar alocações
      const alocacoes = await Promise.all(
        validatedData.alocacoes.map(alocacao =>
          prisma.alocacao.create({
            data: {
              resultado_id: novoResultado.id,
              sala_id: alocacao.sala_id,
              turma_id: alocacao.turma_id,
              compatibilidade_score: alocacao.compatibilidade_score,
              observacoes: alocacao.observacoes
            }
          })
        )
      );

      // Atualizar status do projeto e data da última alocação
      await prisma.projetoAlocacao.update({
        where: { id: validatedData.projeto_id },
        data: {
          status: 'ALOCADO',
          ultima_alocacao: new Date()
        }
      });

      return { ...novoResultado, alocacoes };
    });

    const response: ApiResponse = {
      success: true,
      data: resultado,
      message: 'Resultado de alocação criado com sucesso'
    };

    res.status(201).json(response);
  } catch (error) {
    console.error('Erro ao criar resultado:', error);
    
    if (error instanceof z.ZodError) {
      return res.status(400).json({
        success: false,
        error: 'Dados inválidos',
        details: error.errors
      });
    }

    res.status(500).json({
      success: false,
      error: 'Erro interno do servidor'
    });
  }
});

// POST /api/resultados/executar/:projetoId - Executar algoritmo de alocação
router.post('/executar/:projetoId', async (req, res) => {
  try {
    const { projetoId } = req.params;
    const { priorizar_capacidade = true, priorizar_especiais = true, priorizar_proximidade = true } = req.body;

    // Buscar projeto com salas e turmas
    const projeto = await req.prisma.projetoAlocacao.findUnique({
      where: { id: projetoId },
      include: {
        salas: {
          include: {
            sala: true
          }
        },
        turmas: {
          include: {
            turma: true
          }
        }
      }
    });

    if (!projeto) {
      return res.status(404).json({
        success: false,
        error: 'Projeto não encontrado'
      });
    }

    if (projeto.salas.length === 0) {
      return res.status(400).json({
        success: false,
        error: 'Projeto deve ter pelo menos uma sala'
      });
    }

    if (projeto.turmas.length === 0) {
      return res.status(400).json({
        success: false,
        error: 'Projeto deve ter pelo menos uma turma'
      });
    }

    // Preparar dados para o script Python
    console.log('🔄 [RESULTADOS] Preparando dados para Python...');
    const dadosParaPython = {
      salas: projeto.salas.map(ps => ps.sala),
      turmas: projeto.turmas.map(pt => pt.turma)
    };

    const parametrosPython = {
      priorizar_capacidade,
      priorizar_especiais,
      priorizar_proximidade
    };

    console.log('📊 [RESULTADOS] Dados preparados:', {
      salasCount: dadosParaPython.salas.length,
      turmasCount: dadosParaPython.turmas.length,
      salasSample: dadosParaPython.salas.slice(0, 2),
      turmasSample: dadosParaPython.turmas.slice(0, 2),
      parametros: parametrosPython
    });

    // Executar algoritmo Python
    console.log('🐍 [RESULTADOS] Chamando serviço Python...');
    const resultadoPython = await pythonService.executarAlocacaoInteligente(
      dadosParaPython,
      parametrosPython
    );
    
    console.log('📥 [RESULTADOS] Resultado do Python recebido:', resultadoPython);

    if (!resultadoPython.success) {
      return res.status(500).json({
        success: false,
        error: `Erro no algoritmo de alocação: ${resultadoPython.error}`
      });
    }

    const alocacoes = resultadoPython.alocacoes || [];
    const scoreOtimizacao = resultadoPython.score_otimizacao || 0;

    // Criar resultado no banco
    const resultado = await req.prisma.$transaction(async (prisma) => {
      const novoResultado = await prisma.resultadoAlocacao.create({
        data: {
          projeto_id: projetoId,
          score_otimizacao: scoreOtimizacao,
          priorizar_capacidade,
          priorizar_especiais,
          priorizar_proximidade
        }
      });

      const alocacoesCriadas = await Promise.all(
        alocacoes.map(alocacao =>
          prisma.alocacao.create({
            data: {
              resultado_id: novoResultado.id,
              ...alocacao
            },
            include: {
              sala: true,
              turma: true
            }
          })
        )
      );

      await prisma.projetoAlocacao.update({
        where: { id: projetoId },
        data: {
          status: 'ALOCADO',
          ultima_alocacao: new Date()
        }
      });

      return {
        ...novoResultado,
        alocacoes: alocacoesCriadas
      };
    });

    const response: ApiResponse = {
      success: true,
      data: resultado,
      message: 'Alocação executada com sucesso'
    };

    res.json(response);
  } catch (error) {
    console.error('Erro ao executar alocação:', error);
    res.status(500).json({
      success: false,
      error: 'Erro interno do servidor'
    });
  }
});

// DELETE /api/resultados/:id - Deletar resultado
router.delete('/:id', async (req, res) => {
  try {
    const { id } = req.params;

    // Verificar se o resultado existe
    const existingResultado = await req.prisma.resultadoAlocacao.findUnique({
      where: { id }
    });

    if (!existingResultado) {
      return res.status(404).json({
        success: false,
        error: 'Resultado não encontrado'
      });
    }

    await req.prisma.resultadoAlocacao.delete({
      where: { id }
    });

    const response: ApiResponse = {
      success: true,
      message: 'Resultado deletado com sucesso'
    };

    res.json(response);
  } catch (error) {
    console.error('Erro ao deletar resultado:', error);
    res.status(500).json({
      success: false,
      error: 'Erro interno do servidor'
    });
  }
});

export default router;
