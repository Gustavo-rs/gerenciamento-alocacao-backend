import { Router } from 'express';
import { TurmaSchema, UpdateTurmaData, ApiResponse } from '../types';
import { z } from 'zod';

const router = Router();

// GET /api/turmas - Listar todas as turmas
router.get('/', async (req, res) => {
  try {
    const turmas = await req.prisma.turma.findMany({
      orderBy: { created_at: 'desc' }
    });

    const response: ApiResponse = {
      success: true,
      data: turmas
    };

    res.json(response);
  } catch (error) {
    console.error('Erro ao buscar turmas:', error);
    res.status(500).json({
      success: false,
      error: 'Erro interno do servidor'
    });
  }
});

// GET /api/turmas/:id - Buscar turma por ID
router.get('/:id', async (req, res) => {
  try {
    const { id } = req.params;

    const turma = await req.prisma.turma.findUnique({
      where: { id },
      include: {
        projetos: {
          include: {
            projeto: true
          }
        }
      }
    });

    if (!turma) {
      return res.status(404).json({
        success: false,
        error: 'Turma não encontrada'
      });
    }

    const response: ApiResponse = {
      success: true,
      data: turma
    };

    res.json(response);
  } catch (error) {
    console.error('Erro ao buscar turma:', error);
    res.status(500).json({
      success: false,
      error: 'Erro interno do servidor'
    });
  }
});

// POST /api/turmas - Criar nova turma
router.post('/', async (req, res) => {
  try {
    const validatedData = TurmaSchema.parse(req.body);

    // Verificar se já existe uma turma com o mesmo id_turma
    const existingTurma = await req.prisma.turma.findUnique({
      where: { id_turma: validatedData.id_turma }
    });

    if (existingTurma) {
      return res.status(400).json({
        success: false,
        error: 'Já existe uma turma com este ID'
      });
    }

    const novaTurma = await req.prisma.turma.create({
      data: validatedData
    });

    const response: ApiResponse = {
      success: true,
      data: novaTurma,
      message: 'Turma criada com sucesso'
    };

    res.status(201).json(response);
  } catch (error) {
    console.error('Erro ao criar turma:', error);
    
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

// PUT /api/turmas/:id - Atualizar turma
router.put('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    const updateData: UpdateTurmaData = req.body;

    // Verificar se a turma existe
    const existingTurma = await req.prisma.turma.findUnique({
      where: { id }
    });

    if (!existingTurma) {
      return res.status(404).json({
        success: false,
        error: 'Turma não encontrada'
      });
    }

    // Validar apenas os campos que foram enviados
    const fieldsToUpdate: any = {};
    
    if (updateData.nome !== undefined) fieldsToUpdate.nome = updateData.nome;
    if (updateData.alunos !== undefined) fieldsToUpdate.alunos = updateData.alunos;
    if (updateData.esp_necessarias !== undefined) fieldsToUpdate.esp_necessarias = updateData.esp_necessarias;

    const turmaAtualizada = await req.prisma.turma.update({
      where: { id },
      data: fieldsToUpdate
    });

    const response: ApiResponse = {
      success: true,
      data: turmaAtualizada,
      message: 'Turma atualizada com sucesso'
    };

    res.json(response);
  } catch (error) {
    console.error('Erro ao atualizar turma:', error);
    res.status(500).json({
      success: false,
      error: 'Erro interno do servidor'
    });
  }
});

// DELETE /api/turmas/:id - Deletar turma
router.delete('/:id', async (req, res) => {
  try {
    const { id } = req.params;

    // Verificar se a turma existe
    const existingTurma = await req.prisma.turma.findUnique({
      where: { id }
    });

    if (!existingTurma) {
      return res.status(404).json({
        success: false,
        error: 'Turma não encontrada'
      });
    }

    await req.prisma.turma.delete({
      where: { id }
    });

    const response: ApiResponse = {
      success: true,
      message: 'Turma deletada com sucesso'
    };

    res.json(response);
  } catch (error) {
    console.error('Erro ao deletar turma:', error);
    res.status(500).json({
      success: false,
      error: 'Erro interno do servidor'
    });
  }
});

export default router;
