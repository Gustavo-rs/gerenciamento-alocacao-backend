import { Router } from 'express';
import { z } from 'zod';
import { PrismaClient } from '@prisma/client';
import { spawn } from 'child_process';
import path from 'path';

const router = Router();
const prisma = new PrismaClient();

// Schema para parâmetros de alocação
const AlocacaoParametrosSchema = z.object({
  priorizar_capacidade: z.boolean().default(true),
  priorizar_especiais: z.boolean().default(true),
  priorizar_proximidade: z.boolean().default(true)
});

// POST /api/alocacao-inteligente/:alocacao_id - Executar alocação inteligente para uma alocação principal
router.post('/:alocacao_id', async (req, res) => {
  try {
    const { alocacao_id } = req.params;
    const parametros = AlocacaoParametrosSchema.parse(req.body);

    console.log(`🚀 Iniciando alocação inteligente para alocação: ${alocacao_id}`);

    // Buscar a alocação principal com todos os dados
    const alocacao = await prisma.alocacaoPrincipal.findUnique({
      where: { id: alocacao_id },
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

    // Verificar se há salas
    if (alocacao.salas.length === 0) {
      return res.status(400).json({
        success: false,
        error: 'A alocação não possui salas associadas'
      });
    }

    // Verificar se há horários com turmas
    const horariosComTurmas = alocacao.horarios.filter(h => h.turmas.length > 0);
    if (horariosComTurmas.length === 0) {
      return res.status(400).json({
        success: false,
        error: 'A alocação não possui horários com turmas'
      });
    }

    console.log(`📊 Processando ${horariosComTurmas.length} horários com turmas`);

    // Processar cada horário separadamente
    const resultadosHorarios = [];
    
    for (const horario of horariosComTurmas) {
      console.log(`⏰ Processando horário: ${horario.dia_semana} ${horario.periodo}`);
      
      // Preparar dados para o Python
      const salasData = alocacao.salas.map(as => ({
        id: as.sala.id,
        id_sala: as.sala.id_sala,
        nome: as.sala.nome,
        capacidade_total: as.sala.capacidade_total,
        localizacao: as.sala.localizacao,
        status: as.sala.status,
        cadeiras_moveis: as.sala.cadeiras_moveis > 0,
        cadeiras_especiais: as.sala.cadeiras_especiais
      }));

      const turmasData = horario.turmas.map(ht => ({
        id: ht.turma.id,
        id_turma: ht.turma.id_turma,
        nome: ht.turma.nome,
        alunos: ht.turma.alunos,
        duracao_min: ht.turma.duracao_min || 60,
        esp_necessarias: ht.turma.esp_necessarias
      }));

      console.log(`📈 Horário ${horario.dia_semana} ${horario.periodo}: ${turmasData.length} turmas, ${salasData.length} salas`);

      // Executar algoritmo Python para este horário
      const resultadoPython = await executarAlgoritmoPython({
        salas: salasData,
        turmas: turmasData
      }, parametros);

      if (resultadoPython.success) {
        // Verificar se já existe resultado para este horário e deletar se existir
        const resultadoExistente = await prisma.resultadoAlocacaoHorario.findUnique({
          where: {
            alocacao_id_horario_id: {
              alocacao_id: alocacao_id,
              horario_id: horario.id
            }
          }
        });

        if (resultadoExistente) {
          await prisma.resultadoAlocacaoHorario.delete({
            where: { id: resultadoExistente.id }
          });
        }

        // Salvar novo resultado no banco
        const resultadoSalvo = await prisma.resultadoAlocacaoHorario.create({
          data: {
            alocacao_id: alocacao_id,
            horario_id: horario.id,
            score_otimizacao: resultadoPython.score_otimizacao,
            acuracia_modelo: resultadoPython.acuracia_modelo,
            total_turmas: resultadoPython.total_turmas || 0,
            turmas_alocadas: resultadoPython.total_alocacoes || 0,
            turmas_sobrando: resultadoPython.turmas_sobrando || 0,
            priorizar_capacidade: parametros.priorizar_capacidade,
            priorizar_especiais: parametros.priorizar_especiais,
            priorizar_proximidade: parametros.priorizar_proximidade,
            analise_detalhada: JSON.stringify(resultadoPython.analise_detalhada),
            debug_info: JSON.stringify(resultadoPython.debug_info || {}),
            turmas_nao_alocadas: JSON.stringify(resultadoPython.turmas_nao_alocadas || []),
            alocacoes: {
              create: resultadoPython.alocacoes.map((alocacao: any) => ({
                sala_id: alocacao.sala_id,
                turma_id: alocacao.turma_id,
                compatibilidade_score: alocacao.compatibilidade_score,
                observacoes: alocacao.observacoes
              }))
            }
          },
          include: {
            alocacoes: {
              include: {
                sala: true,
                turma: true
              }
            },
            horario: true
          }
        });

        resultadosHorarios.push({
          horario: {
            id: horario.id,
            dia_semana: horario.dia_semana,
            periodo: horario.periodo
          },
          resultado: resultadoSalvo,
          python_result: resultadoPython
        });

        console.log(`✅ Horário ${horario.dia_semana} ${horario.periodo} processado com sucesso`);
      } else {
        console.log(`❌ Erro no horário ${horario.dia_semana} ${horario.periodo}: ${resultadoPython.error}`);
        
        // Salvar erro no banco para histórico
        try {
          const resultadoErro = await prisma.resultadoAlocacaoHorario.create({
            data: {
              alocacao_id: alocacao_id,
              horario_id: horario.id,
              score_otimizacao: 0,
              acuracia_modelo: 0,
              total_turmas: turmasData.length,
              turmas_alocadas: 0,
              turmas_sobrando: turmasData.length,
              priorizar_capacidade: parametros.priorizar_capacidade,
              priorizar_especiais: parametros.priorizar_especiais,
              priorizar_proximidade: parametros.priorizar_proximidade,
              analise_detalhada: JSON.stringify({ erro: true, detalhes: resultadoPython.error }),
              debug_info: JSON.stringify(resultadoPython.diagnostico || {}),
              turmas_nao_alocadas: JSON.stringify(turmasData.map(t => ({
                id: t.id,
                nome: t.nome,
                alunos: t.alunos,
                esp_necessarias: t.esp_necessarias,
                motivo: "Erro no processamento da alocação"
              }))),
              alocacoes: { create: [] }
            }
          });
          
          resultadosHorarios.push({
            horario: {
              id: horario.id,
              dia_semana: horario.dia_semana,
              periodo: horario.periodo
            },
            resultado: resultadoErro,
            erro: resultadoPython.error,
            python_result: {
              success: false,
              error: resultadoPython.error,
              total_turmas: turmasData.length,
              turmas_alocadas: 0,
              turmas_sobrando: turmasData.length,
              score_otimizacao: 0
            }
          });
        } catch (dbError) {
          console.error(`Erro ao salvar resultado de erro no banco:`, dbError);
          resultadosHorarios.push({
            horario: {
              id: horario.id,
              dia_semana: horario.dia_semana,
              periodo: horario.periodo
            },
            erro: resultadoPython.error
          });
        }
      }
    }

    // Calcular estatísticas gerais
    const sucessos = resultadosHorarios.filter(r => !r.erro);
    const scoreGeral = sucessos.length > 0 
      ? sucessos.reduce((acc, r) => acc + r.python_result.score_otimizacao, 0) / sucessos.length
      : 0;

    console.log(`🎯 Alocação inteligente concluída: ${sucessos.length}/${resultadosHorarios.length} horários processados com sucesso`);

    res.json({
      success: true,
      message: `Alocação inteligente executada para ${sucessos.length} de ${resultadosHorarios.length} horários`,
      dados: {
        alocacao_id: alocacao_id,
        total_horarios: resultadosHorarios.length,
        horarios_processados: sucessos.length,
        score_geral: Math.round(scoreGeral * 100) / 100,
        resultados_por_horario: resultadosHorarios
      }
    });

  } catch (error) {
    console.error('Erro na alocação inteligente:', error);
    
    if (error instanceof z.ZodError) {
      return res.status(400).json({
        success: false,
        error: 'Parâmetros inválidos',
        details: error.errors
      });
    }

    res.status(500).json({
      success: false,
      error: 'Erro interno do servidor'
    });
  }
});

// GET /api/alocacao-inteligente/:alocacao_id/resultados - Buscar resultados da alocação
router.get('/:alocacao_id/resultados', async (req, res) => {
  try {
    const { alocacao_id } = req.params;

    const resultados = await prisma.resultadoAlocacaoHorario.findMany({
      where: { alocacao_id },
      include: {
        horario: true,
        alocacoes: {
          include: {
            sala: true,
            turma: true
          }
        }
      },
      orderBy: [
        { horario: { dia_semana: 'asc' } },
        { horario: { periodo: 'asc' } }
      ]
    });

    res.json({
      success: true,
      data: resultados
    });

  } catch (error) {
    console.error('Erro ao buscar resultados:', error);
    res.status(500).json({
      success: false,
      error: 'Erro interno do servidor'
    });
  }
});

// Função auxiliar para executar o algoritmo Python
async function executarAlgoritmoPython(dados: any, parametros: any): Promise<any> {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(__dirname, '../../scripts/alocacao_inteligente.py');
    
    const dadosJson = JSON.stringify(dados);
    const parametrosJson = JSON.stringify(parametros);
    
    console.log(`🐍 Executando Python script: ${scriptPath}`);
    
    const pythonProcess = spawn('python', [
      scriptPath,
      '--dados', dadosJson,
      '--parametros', parametrosJson
    ]);
    
    let stdout = '';
    let stderr = '';
    
    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
    });
    
    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
      console.log(`🐍 [Python stderr]: ${data.toString()}`);
    });
    
    pythonProcess.on('close', (code) => {
      console.log(`🐍 Python process finished with code: ${code}`);
      
      if (code !== 0) {
        console.error(`❌ Python script failed: ${stderr}`);
        resolve({
          success: false,
          error: `Algoritmo Python falhou: ${stderr}`,
          alocacoes: [],
          score_otimizacao: 0
        });
        return;
      }
      
      try {
        const resultado = JSON.parse(stdout);
        console.log(`✅ Python result parsed successfully`);
        resolve(resultado);
      } catch (error) {
        console.error(`❌ Failed to parse Python output: ${error}`);
        console.error(`📝 Raw output: ${stdout}`);
        resolve({
          success: false,
          error: `Erro ao processar resultado do algoritmo: ${error}`,
          alocacoes: [],
          score_otimizacao: 0
        });
      }
    });
    
    pythonProcess.on('error', (error) => {
      console.error(`❌ Python process error: ${error}`);
      resolve({
        success: false,
        error: `Erro ao executar algoritmo Python: ${error.message}`,
        alocacoes: [],
        score_otimizacao: 0
      });
    });
  });
}

export default router;
