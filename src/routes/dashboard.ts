import { Router } from 'express';
import { PrismaClient } from '@prisma/client';

const router = Router();
const prisma = new PrismaClient();

// Obter estatísticas para o dashboard
router.get('/stats', async (req, res) => {
  try {
    // Estatísticas das alocações principais
    const totalAlocacoes = await prisma.alocacaoPrincipal.count();
    
    // Estatísticas das salas
    const totalSalas = await prisma.sala.count();
    const salasAtivas = await prisma.sala.count({
      where: { status: 'ATIVA' }
    });

    // Estatísticas das turmas
    const totalTurmas = await prisma.turma.count();
    const somaAlunos = await prisma.turma.aggregate({
      _sum: { alunos: true }
    });

    // Capacidade total das salas
    const capacidadeTotal = await prisma.sala.aggregate({
      _sum: { capacidade_total: true }
    });

    // Cadeiras especiais
    const cadeirasEspeciais = await prisma.sala.aggregate({
      _sum: { cadeiras_especiais: true }
    });

    // Resultados de alocação inteligente gerados
    const resultadosGerados = await prisma.resultadoAlocacaoHorario.count();

    // Alocações com horários
    const alocacoesComHorarios = await prisma.alocacaoPrincipal.findMany({
      include: {
        _count: {
          select: {
            horarios: true,
            salas: true
          }
        }
      }
    });

    // Alocações ativas (que têm salas e horários)
    const alocacoesAtivas = alocacoesComHorarios.filter(
      a => a._count.salas > 0 && a._count.horarios > 0
    ).length;

    // Alocações problemáticas (sem salas ou sem horários)
    const alocacoesProblematicas = alocacoesComHorarios.filter(
      a => a._count.salas === 0 || a._count.horarios === 0
    );

    res.json({
      success: true,
      data: {
        stats: {
          totalAlocacoes,
          alocacoesAtivas,
          totalSalas,
          salasAtivas,
          totalTurmas,
          totalAlunos: somaAlunos._sum.alunos || 0,
          capacidadeTotal: capacidadeTotal._sum.capacidade_total || 0,
          cadeirasEspeciais: cadeirasEspeciais._sum.cadeiras_especiais || 0,
          resultadosGerados
        },
        alocacoesProblematicas: alocacoesProblematicas.map(a => ({
          id: a.id,
          nome: a.nome,
          salas: a._count.salas,
          horarios: a._count.horarios
        })),
        alocacoesCompatibilidade: alocacoesComHorarios.map(a => ({
          id: a.id,
          nome: a.nome,
          salas: a._count.salas,
          horarios: a._count.horarios,
          temResultados: false // TODO: implementar verificação de resultados
        }))
      }
    });
  } catch (error) {
    console.error('Erro ao obter estatísticas:', error);
    res.status(500).json({
      success: false,
      error: 'Erro interno do servidor'
    });
  }
});

// Obter alocações recentes
router.get('/recentes', async (req, res) => {
  try {
    const alocacoesRecentes = await prisma.alocacaoPrincipal.findMany({
      orderBy: { created_at: 'desc' },
      take: 5,
      include: {
        _count: {
          select: {
            salas: true,
            horarios: true
          }
        }
      }
    });

    res.json({
      success: true,
      data: alocacoesRecentes.map(a => ({
        id: a.id,
        nome: a.nome,
        descricao: a.descricao,
        salas: a._count.salas,
        horarios: a._count.horarios,
        created_at: a.created_at
      }))
    });
  } catch (error) {
    console.error('Erro ao obter alocações recentes:', error);
    res.status(500).json({
      success: false,
      error: 'Erro interno do servidor'
    });
  }
});

export default router;
