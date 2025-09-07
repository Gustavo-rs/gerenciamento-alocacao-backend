import { Router } from 'express';
import { SalaSchema, UpdateSalaData, ApiResponse } from '../types';
import { z } from 'zod';

const router = Router();

// GET /api/salas - Listar todas as salas
router.get('/', async (req, res) => {
  try {
    const salas = await req.prisma.sala.findMany({
      orderBy: { created_at: 'desc' }
    });

    const response: ApiResponse = {
      success: true,
      data: salas
    };

    res.json(response);
  } catch (error) {
    console.error('Erro ao buscar salas:', error);
    res.status(500).json({
      success: false,
      error: 'Erro interno do servidor'
    });
  }
});

// GET /api/salas/:id - Buscar sala por ID
router.get('/:id', async (req, res) => {
  try {
    const { id } = req.params;

    const sala = await req.prisma.sala.findUnique({
      where: { id },
      include: {
        projetos: {
          include: {
            projeto: true
          }
        }
      }
    });

    if (!sala) {
      return res.status(404).json({
        success: false,
        error: 'Sala não encontrada'
      });
    }

    const response: ApiResponse = {
      success: true,
      data: sala
    };

    res.json(response);
  } catch (error) {
    console.error('Erro ao buscar sala:', error);
    res.status(500).json({
      success: false,
      error: 'Erro interno do servidor'
    });
  }
});

// POST /api/salas - Criar nova sala
router.post('/', async (req, res) => {
  try {
    const validatedData = SalaSchema.parse(req.body);

    // Verificar se já existe uma sala com o mesmo id_sala
    const existingSala = await req.prisma.sala.findUnique({
      where: { id_sala: validatedData.id_sala }
    });

    if (existingSala) {
      return res.status(400).json({
        success: false,
        error: 'Já existe uma sala com este ID'
      });
    }

    const novaSala = await req.prisma.sala.create({
      data: validatedData
    });

    const response: ApiResponse = {
      success: true,
      data: novaSala,
      message: 'Sala criada com sucesso'
    };

    res.status(201).json(response);
  } catch (error) {
    console.error('Erro ao criar sala:', error);
    
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

// PUT /api/salas/:id - Atualizar sala
router.put('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    const updateData: UpdateSalaData = req.body;

    // Verificar se a sala existe
    const existingSala = await req.prisma.sala.findUnique({
      where: { id }
    });

    if (!existingSala) {
      return res.status(404).json({
        success: false,
        error: 'Sala não encontrada'
      });
    }

    // Validar apenas os campos que foram enviados
    const fieldsToUpdate: any = {};
    
    if (updateData.nome !== undefined) fieldsToUpdate.nome = updateData.nome;
    if (updateData.capacidade_total !== undefined) fieldsToUpdate.capacidade_total = updateData.capacidade_total;
    if (updateData.localizacao !== undefined) fieldsToUpdate.localizacao = updateData.localizacao;
    if (updateData.status !== undefined) fieldsToUpdate.status = updateData.status;
    if (updateData.cadeiras_moveis !== undefined) fieldsToUpdate.cadeiras_moveis = updateData.cadeiras_moveis;
    if (updateData.cadeiras_especiais !== undefined) fieldsToUpdate.cadeiras_especiais = updateData.cadeiras_especiais;

    const salaAtualizada = await req.prisma.sala.update({
      where: { id },
      data: fieldsToUpdate
    });

    const response: ApiResponse = {
      success: true,
      data: salaAtualizada,
      message: 'Sala atualizada com sucesso'
    };

    res.json(response);
  } catch (error) {
    console.error('Erro ao atualizar sala:', error);
    res.status(500).json({
      success: false,
      error: 'Erro interno do servidor'
    });
  }
});

// DELETE /api/salas/:id - Deletar sala
router.delete('/:id', async (req, res) => {
  try {
    const { id } = req.params;

    // Verificar se a sala existe
    const existingSala = await req.prisma.sala.findUnique({
      where: { id }
    });

    if (!existingSala) {
      return res.status(404).json({
        success: false,
        error: 'Sala não encontrada'
      });
    }

    await req.prisma.sala.delete({
      where: { id }
    });

    const response: ApiResponse = {
      success: true,
      message: 'Sala deletada com sucesso'
    };

    res.json(response);
  } catch (error) {
    console.error('Erro ao deletar sala:', error);
    res.status(500).json({
      success: false,
      error: 'Erro interno do servidor'
    });
  }
});

export default router;
