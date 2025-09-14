import { Router } from 'express';
import { z } from 'zod';
import { PrismaClient } from '@prisma/client';

const router = Router();
const prisma = new PrismaClient();

// Schema para criar/atualizar alocação
const AlocacaoSchema = z.object({
  nome: z.string().min(1, 'Nome é obrigatório'),
  descricao: z.string().min(1, 'Descrição é obrigatória')
});

// GET /api/alocacoes - Listar todas as alocações
router.get('/', async (req, res) => {
  try {
    const alocacoes = await prisma.alocacaoPrincipal.findMany({
      include: {
        salas: {
          include: {
            sala: true
          }
        },
        horarios: {
          include: {
            turmas: {
              include: {
                turma: true
              }
            }
          }
        }
      },
      orderBy: {
        created_at: 'desc'
      }
    });

    res.json({
      success: true,
      data: alocacoes
    });
  } catch (error) {
    console.error('Erro ao buscar alocações:', error);
    res.status(500).json({
      success: false,
      error: 'Erro interno do servidor'
    });
  }
});

// GET /api/alocacoes/:id - Buscar alocação por ID
router.get('/:id', async (req, res) => {
  try {
    const { id } = req.params;

    const alocacao = await prisma.alocacaoPrincipal.findUnique({
      where: { id },
      include: {
        salas: {
          include: {
            sala: true
          }
        },
        horarios: {
          include: {
            turmas: {
              include: {
                turma: true
              }
            }
          }
        }
      }
    });

    if (!alocacao) {
      return res.status(404).json({
        success: false,
        error: 'Alocação não encontrada'
      });
    }

    res.json({
      success: true,
      data: alocacao
    });
  } catch (error) {
    console.error('Erro ao buscar alocação:', error);
    res.status(500).json({
      success: false,
      error: 'Erro interno do servidor'
    });
  }
});

// POST /api/alocacoes - Criar nova alocação
router.post('/', async (req, res) => {
  try {
    const validatedData = AlocacaoSchema.parse(req.body);

    // Verificar se já existe uma alocação com o mesmo nome
    const existingAlocacao = await prisma.alocacaoPrincipal.findFirst({
      where: { nome: validatedData.nome }
    });

    if (existingAlocacao) {
      return res.status(400).json({
        success: false,
        error: 'Já existe uma alocação com este nome'
      });
    }

    const alocacao = await prisma.alocacaoPrincipal.create({
      data: {
        nome: validatedData.nome,
        descricao: validatedData.descricao
      },
      include: {
        salas: {
          include: {
            sala: true
          }
        }
      }
    });

    res.status(201).json({
      success: true,
      data: alocacao
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return res.status(400).json({
        success: false,
        error: 'Dados inválidos',
        details: error.errors
      });
    }

    console.error('Erro ao criar alocação:', error);
    res.status(500).json({
      success: false,
      error: 'Erro interno do servidor'
    });
  }
});

// PUT /api/alocacoes/:id - Atualizar alocação
router.put('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    const validatedData = AlocacaoSchema.parse(req.body);

    const alocacao = await prisma.alocacaoPrincipal.update({
      where: { id },
      data: {
        nome: validatedData.nome,
        descricao: validatedData.descricao
      },
      include: {
        salas: {
          include: {
            sala: true
          }
        }
      }
    });

    res.json({
      success: true,
      data: alocacao
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return res.status(400).json({
        success: false,
        error: 'Dados inválidos',
        details: error.errors
      });
    }

    console.error('Erro ao atualizar alocação:', error);
    res.status(500).json({
      success: false,
      error: 'Erro interno do servidor'
    });
  }
});

// DELETE /api/alocacoes/:id - Excluir alocação
router.delete('/:id', async (req, res) => {
  try {
    const { id } = req.params;

    // Verificar se a alocação existe e contar elementos relacionados para o log
    const alocacao = await prisma.alocacaoPrincipal.findUnique({
      where: { id },
      include: {
        salas: true,
        horarios: {
          include: {
            turmas: true
          }
        }
      }
    });

    if (!alocacao) {
      return res.status(404).json({
        success: false,
        error: 'Alocação não encontrada'
      });
    }

    // Contar elementos que serão deletados
    const salasCount = alocacao.salas.length;
    const horariosCount = alocacao.horarios.length;
    const turmasCount = alocacao.horarios.reduce((acc, h) => acc + h.turmas.length, 0);

    console.log(`Excluindo alocação "${alocacao.nome}" e seus relacionamentos:`);
    console.log(`- ${salasCount} salas associadas`);
    console.log(`- ${horariosCount} horários`);
    console.log(`- ${turmasCount} turmas associadas aos horários`);

    // O Prisma automaticamente excluirá em cascata:
    // - AlocacaoSala (pela foreign key com onDelete: Cascade)
    // - Horario (pela foreign key com onDelete: Cascade)
    // - HorarioTurma (pela foreign key de Horario com onDelete: Cascade)
    await prisma.alocacaoPrincipal.delete({
      where: { id }
    });

    res.json({
      success: true,
      message: `Alocação excluída com sucesso! Foram removidos: ${salasCount} salas, ${horariosCount} horários e ${turmasCount} turmas associadas.`
    });
  } catch (error) {
    console.error('Erro ao excluir alocação:', error);
    res.status(500).json({
      success: false,
      error: 'Erro interno do servidor'
    });
  }
});

// POST /api/alocacoes/:id/salas - Adicionar sala à alocação
router.post('/:id/salas', async (req, res) => {
  try {
    const { id } = req.params;
    const { sala_id } = req.body;

    if (!sala_id) {
      return res.status(400).json({
        success: false,
        error: 'ID da sala é obrigatório'
      });
    }

    // Verificar se a alocação existe
    const alocacao = await prisma.alocacaoPrincipal.findUnique({
      where: { id }
    });

    if (!alocacao) {
      return res.status(404).json({
        success: false,
        error: 'Alocação não encontrada'
      });
    }

    // Verificar se a sala existe
    const sala = await prisma.sala.findUnique({
      where: { id: sala_id }
    });

    if (!sala) {
      return res.status(404).json({
        success: false,
        error: 'Sala não encontrada'
      });
    }

    // Verificar se a associação já existe
    const existingAssociation = await prisma.alocacaoSala.findUnique({
      where: {
        alocacao_id_sala_id: {
          alocacao_id: id,
          sala_id: sala_id
        }
      }
    });

    if (existingAssociation) {
      return res.status(400).json({
        success: false,
        error: 'Sala já está adicionada a esta alocação'
      });
    }

    // Criar a associação
    await prisma.alocacaoSala.create({
      data: {
        alocacao_id: id,
        sala_id: sala_id
      }
    });

    res.status(201).json({
      success: true,
      message: 'Sala adicionada à alocação com sucesso'
    });
  } catch (error) {
    console.error('Erro ao adicionar sala à alocação:', error);
    res.status(500).json({
      success: false,
      error: 'Erro interno do servidor'
    });
  }
});

// DELETE /api/alocacoes/:id/salas/:sala_id - Remover sala da alocação
router.delete('/:id/salas/:sala_id', async (req, res) => {
  try {
    const { id, sala_id } = req.params;

    // Verificar se a associação existe
    const association = await prisma.alocacaoSala.findUnique({
      where: {
        alocacao_id_sala_id: {
          alocacao_id: id,
          sala_id: sala_id
        }
      }
    });

    if (!association) {
      return res.status(404).json({
        success: false,
        error: 'Associação não encontrada'
      });
    }

    // Remover a associação
    await prisma.alocacaoSala.delete({
      where: {
        alocacao_id_sala_id: {
          alocacao_id: id,
          sala_id: sala_id
        }
      }
    });

    res.json({
      success: true,
      message: 'Sala removida da alocação com sucesso'
    });
  } catch (error) {
    console.error('Erro ao remover sala da alocação:', error);
    res.status(500).json({
      success: false,
      error: 'Erro interno do servidor'
    });
  }
});

// POST /api/alocacoes/:id/horarios - Adicionar horário à alocação
router.post('/:id/horarios', async (req, res) => {
  try {
    const { id } = req.params;
    const { dia_semana, periodo } = req.body;

    if (!dia_semana || !periodo) {
      return res.status(400).json({
        success: false,
        error: 'Dia da semana e período são obrigatórios'
      });
    }

    // Verificar se a alocação existe
    const alocacao = await prisma.alocacaoPrincipal.findUnique({
      where: { id }
    });

    if (!alocacao) {
      return res.status(404).json({
        success: false,
        error: 'Alocação não encontrada'
      });
    }

    // Verificar se já existe horário para esse dia/período
    const existingHorario = await prisma.horario.findUnique({
      where: {
        alocacao_id_dia_semana_periodo: {
          alocacao_id: id,
          dia_semana,
          periodo
        }
      }
    });

    if (existingHorario) {
      return res.status(400).json({
        success: false,
        error: 'Já existe um horário para este dia e período'
      });
    }

    // Criar o horário
    const horario = await prisma.horario.create({
      data: {
        alocacao_id: id,
        dia_semana,
        periodo
      },
      include: {
        turmas: {
          include: {
            turma: true
          }
        }
      }
    });

    res.status(201).json({
      success: true,
      data: horario
    });
  } catch (error) {
    console.error('Erro ao criar horário:', error);
    res.status(500).json({
      success: false,
      error: 'Erro interno do servidor'
    });
  }
});

export default router;
