import { Router } from 'express';
import { z } from 'zod';
import { PrismaClient } from '@prisma/client';

const router = Router();
const prisma = new PrismaClient();

// DELETE /api/horarios/:id - Excluir horário
router.delete('/:id', async (req, res) => {
  try {
    const { id } = req.params;

    // Verificar se o horário existe
    const horario = await prisma.horario.findUnique({
      where: { id }
    });

    if (!horario) {
      return res.status(404).json({
        success: false,
        error: 'Horário não encontrado'
      });
    }

    // Primeiro, excluir as associações com turmas
    await prisma.horarioTurma.deleteMany({
      where: { horario_id: id }
    });

    // Depois, excluir o horário
    await prisma.horario.delete({
      where: { id }
    });

    res.json({
      success: true,
      message: 'Horário excluído com sucesso'
    });
  } catch (error) {
    console.error('Erro ao excluir horário:', error);
    res.status(500).json({
      success: false,
      error: 'Erro interno do servidor'
    });
  }
});

// POST /api/horarios/:id/turmas - Adicionar turma ao horário
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

    // Verificar se o horário existe
    const horario = await prisma.horario.findUnique({
      where: { id }
    });

    if (!horario) {
      return res.status(404).json({
        success: false,
        error: 'Horário não encontrado'
      });
    }

    // Verificar se a turma existe
    const turma = await prisma.turma.findUnique({
      where: { id: turma_id }
    });

    if (!turma) {
      return res.status(404).json({
        success: false,
        error: 'Turma não encontrada'
      });
    }

    // Verificar se a associação já existe
    const existingAssociation = await prisma.horarioTurma.findUnique({
      where: {
        horario_id_turma_id: {
          horario_id: id,
          turma_id: turma_id
        }
      }
    });

    if (existingAssociation) {
      return res.status(400).json({
        success: false,
        error: 'Turma já está adicionada a este horário'
      });
    }

    // Verificar se já existe uma turma com o mesmo nome neste horário
    const turmasNoHorario = await prisma.horarioTurma.findMany({
      where: { horario_id: id },
      include: { turma: true }
    });

    const turmaComMesmoNome = turmasNoHorario.find(ht => ht.turma.nome === turma.nome);
    if (turmaComMesmoNome) {
      return res.status(400).json({
        success: false,
        error: `Já existe uma turma com o nome "${turma.nome}" neste horário`
      });
    }

    // Criar a associação
    await prisma.horarioTurma.create({
      data: {
        horario_id: id,
        turma_id: turma_id
      }
    });

    res.status(201).json({
      success: true,
      message: 'Turma adicionada ao horário com sucesso'
    });
  } catch (error) {
    console.error('Erro ao adicionar turma ao horário:', error);
    res.status(500).json({
      success: false,
      error: 'Erro interno do servidor'
    });
  }
});

// DELETE /api/horarios/:id/turmas/:turma_id - Remover turma do horário
router.delete('/:id/turmas/:turma_id', async (req, res) => {
  try {
    const { id, turma_id } = req.params;

    // Verificar se a associação existe
    const association = await prisma.horarioTurma.findUnique({
      where: {
        horario_id_turma_id: {
          horario_id: id,
          turma_id: turma_id
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
    await prisma.horarioTurma.delete({
      where: {
        horario_id_turma_id: {
          horario_id: id,
          turma_id: turma_id
        }
      }
    });

    res.json({
      success: true,
      message: 'Turma removida do horário com sucesso'
    });
  } catch (error) {
    console.error('Erro ao remover turma do horário:', error);
    res.status(500).json({
      success: false,
      error: 'Erro interno do servidor'
    });
  }
});

export default router;
