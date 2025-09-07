import { Router } from 'express';
import { ProjetoAlocacaoSchema, UpdateProjetoAlocacaoData, ApiResponse } from '../types';
import { z } from 'zod';

const router = Router();

// GET /api/projetos - Listar todos os projetos
router.get('/', async (req, res) => {
  try {
    const projetos = await req.prisma.projetoAlocacao.findMany({
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
        },
        resultados: true
      },
      orderBy: { created_at: 'desc' }
    });

    const response: ApiResponse = {
      success: true,
      data: projetos
    };

    res.json(response);
  } catch (error) {
    console.error('Erro ao buscar projetos:', error);
    res.status(500).json({
      success: false,
      error: 'Erro interno do servidor'
    });
  }
});

// GET /api/projetos/:id - Buscar projeto por ID
router.get('/:id', async (req, res) => {
  try {
    const { id } = req.params;

    const projeto = await req.prisma.projetoAlocacao.findUnique({
      where: { id },
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
        },
        resultados: {
          include: {
            alocacoes: {
              include: {
                sala: true,
                turma: true
              }
            }
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

    const response: ApiResponse = {
      success: true,
      data: projeto
    };

    res.json(response);
  } catch (error) {
    console.error('Erro ao buscar projeto:', error);
    res.status(500).json({
      success: false,
      error: 'Erro interno do servidor'
    });
  }
});

// POST /api/projetos - Criar novo projeto
router.post('/', async (req, res) => {
  try {
    const validatedData = ProjetoAlocacaoSchema.parse(req.body);

    // Verificar se já existe um projeto com o mesmo id_projeto
    const existingProjeto = await req.prisma.projetoAlocacao.findUnique({
      where: { id_projeto: validatedData.id_projeto }
    });

    if (existingProjeto) {
      return res.status(400).json({
        success: false,
        error: 'Já existe um projeto com este ID'
      });
    }

    const novoProjeto = await req.prisma.projetoAlocacao.create({
      data: validatedData,
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

    const response: ApiResponse = {
      success: true,
      data: novoProjeto,
      message: 'Projeto criado com sucesso'
    };

    res.status(201).json(response);
  } catch (error) {
    console.error('Erro ao criar projeto:', error);
    
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

// PUT /api/projetos/:id - Atualizar projeto
router.put('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    const updateData: UpdateProjetoAlocacaoData = req.body;

    // Verificar se o projeto existe
    const existingProjeto = await req.prisma.projetoAlocacao.findUnique({
      where: { id }
    });

    if (!existingProjeto) {
      return res.status(404).json({
        success: false,
        error: 'Projeto não encontrado'
      });
    }

    // Validar apenas os campos que foram enviados
    const fieldsToUpdate: any = {};
    
    if (updateData.nome !== undefined) fieldsToUpdate.nome = updateData.nome;
    if (updateData.descricao !== undefined) fieldsToUpdate.descricao = updateData.descricao;
    if (updateData.status !== undefined) fieldsToUpdate.status = updateData.status;

    const projetoAtualizado = await req.prisma.projetoAlocacao.update({
      where: { id },
      data: fieldsToUpdate,
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

    const response: ApiResponse = {
      success: true,
      data: projetoAtualizado,
      message: 'Projeto atualizado com sucesso'
    };

    res.json(response);
  } catch (error) {
    console.error('Erro ao atualizar projeto:', error);
    res.status(500).json({
      success: false,
      error: 'Erro interno do servidor'
    });
  }
});

// POST /api/projetos/:id/salas - Adicionar sala ao projeto
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

    // Verificar se o projeto existe
    const projeto = await req.prisma.projetoAlocacao.findUnique({
      where: { id }
    });

    if (!projeto) {
      return res.status(404).json({
        success: false,
        error: 'Projeto não encontrado'
      });
    }

    // Verificar se a sala existe
    const sala = await req.prisma.sala.findUnique({
      where: { id: sala_id }
    });

    if (!sala) {
      return res.status(404).json({
        success: false,
        error: 'Sala não encontrada'
      });
    }

    // Verificar se a sala já está no projeto
    const existingAssociation = await req.prisma.projetoSala.findUnique({
      where: {
        projeto_id_sala_id: {
          projeto_id: id,
          sala_id: sala_id
        }
      }
    });

    if (existingAssociation) {
      return res.status(400).json({
        success: false,
        error: 'Sala já está associada a este projeto'
      });
    }

    await req.prisma.projetoSala.create({
      data: {
        projeto_id: id,
        sala_id: sala_id
      }
    });

    const response: ApiResponse = {
      success: true,
      message: 'Sala adicionada ao projeto com sucesso'
    };

    res.json(response);
  } catch (error) {
    console.error('Erro ao adicionar sala ao projeto:', error);
    res.status(500).json({
      success: false,
      error: 'Erro interno do servidor'
    });
  }
});

// DELETE /api/projetos/:id/salas/:salaId - Remover sala do projeto
router.delete('/:id/salas/:salaId', async (req, res) => {
  try {
    const { id, salaId } = req.params;

    await req.prisma.projetoSala.delete({
      where: {
        projeto_id_sala_id: {
          projeto_id: id,
          sala_id: salaId
        }
      }
    });

    const response: ApiResponse = {
      success: true,
      message: 'Sala removida do projeto com sucesso'
    };

    res.json(response);
  } catch (error) {
    console.error('Erro ao remover sala do projeto:', error);
    res.status(500).json({
      success: false,
      error: 'Erro interno do servidor'
    });
  }
});

// POST /api/projetos/:id/turmas - Adicionar turma ao projeto
router.post('/:id/turmas', async (req, res) => {
  try {
    const { id } = req.params;
    const { turma_id } = req.body;

    if (!turma_id) {
      return res.status(400).json({
        success: false,
        error: 'ID da turma é obrigatório'
      });
    }

    // Verificar se o projeto existe
    const projeto = await req.prisma.projetoAlocacao.findUnique({
      where: { id }
    });

    if (!projeto) {
      return res.status(404).json({
        success: false,
        error: 'Projeto não encontrado'
      });
    }

    // Verificar se a turma existe
    const turma = await req.prisma.turma.findUnique({
      where: { id: turma_id }
    });

    if (!turma) {
      return res.status(404).json({
        success: false,
        error: 'Turma não encontrada'
      });
    }

    // Verificar se a turma já está no projeto
    const existingAssociation = await req.prisma.projetoTurma.findUnique({
      where: {
        projeto_id_turma_id: {
          projeto_id: id,
          turma_id: turma_id
        }
      }
    });

    if (existingAssociation) {
      return res.status(400).json({
        success: false,
        error: 'Turma já está associada a este projeto'
      });
    }

    await req.prisma.projetoTurma.create({
      data: {
        projeto_id: id,
        turma_id: turma_id
      }
    });

    const response: ApiResponse = {
      success: true,
      message: 'Turma adicionada ao projeto com sucesso'
    };

    res.json(response);
  } catch (error) {
    console.error('Erro ao adicionar turma ao projeto:', error);
    res.status(500).json({
      success: false,
      error: 'Erro interno do servidor'
    });
  }
});

// DELETE /api/projetos/:id/turmas/:turmaId - Remover turma do projeto
router.delete('/:id/turmas/:turmaId', async (req, res) => {
  try {
    const { id, turmaId } = req.params;

    await req.prisma.projetoTurma.delete({
      where: {
        projeto_id_turma_id: {
          projeto_id: id,
          turma_id: turmaId
        }
      }
    });

    const response: ApiResponse = {
      success: true,
      message: 'Turma removida do projeto com sucesso'
    };

    res.json(response);
  } catch (error) {
    console.error('Erro ao remover turma do projeto:', error);
    res.status(500).json({
      success: false,
      error: 'Erro interno do servidor'
    });
  }
});

// DELETE /api/projetos/:id - Deletar projeto
router.delete('/:id', async (req, res) => {
  try {
    const { id } = req.params;

    // Verificar se o projeto existe
    const existingProjeto = await req.prisma.projetoAlocacao.findUnique({
      where: { id }
    });

    if (!existingProjeto) {
      return res.status(404).json({
        success: false,
        error: 'Projeto não encontrado'
      });
    }

    await req.prisma.projetoAlocacao.delete({
      where: { id }
    });

    const response: ApiResponse = {
      success: true,
      message: 'Projeto deletado com sucesso'
    };

    res.json(response);
  } catch (error) {
    console.error('Erro ao deletar projeto:', error);
    res.status(500).json({
      success: false,
      error: 'Erro interno do servidor'
    });
  }
});

export default router;
