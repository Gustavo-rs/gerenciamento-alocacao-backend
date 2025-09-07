import { spawn } from 'child_process';
import path from 'path';

interface PythonScriptResult {
  success: boolean;
  alocacoes?: any[];
  score_otimizacao?: number;
  total_alocacoes?: number;
  error?: string;
}

interface DadosAlocacao {
  salas: any[];
  turmas: any[];
}

interface ParametrosAlocacao {
  priorizar_capacidade: boolean;
  priorizar_especiais: boolean;
  priorizar_proximidade: boolean;
}

export class PythonService {
  private scriptPath: string;

  constructor() {
    this.scriptPath = path.join(__dirname, '../../scripts/alocacao_inteligente.py');
    console.log('📁 [PYTHON SERVICE] Script path configurado:', this.scriptPath);
    
    // Verificar se o arquivo existe
    const fs = require('fs');
    if (fs.existsSync(this.scriptPath)) {
      console.log('✅ [PYTHON SERVICE] Script Python encontrado!');
    } else {
      console.error('❌ [PYTHON SERVICE] Script Python NÃO encontrado!');
    }
  }

  async executarAlocacaoInteligente(
    dados: DadosAlocacao,
    parametros: ParametrosAlocacao
  ): Promise<PythonScriptResult> {
    console.log('🐍 [PYTHON SERVICE] Iniciando execução do algoritmo Python...');
    console.log('📊 [PYTHON SERVICE] Dados recebidos:', {
      salas: dados.salas?.length || 0,
      turmas: dados.turmas?.length || 0
    });
    
    return new Promise((resolve, reject) => {
      try {
        // Verificar se Python está disponível
        const pythonCommand = process.platform === 'win32' ? 'python' : 'python3';
        console.log(`🔧 [PYTHON SERVICE] Comando Python: ${pythonCommand}`);
        console.log(`📁 [PYTHON SERVICE] Script path: ${this.scriptPath}`);
        
        // Preparar argumentos
        const dadosJson = JSON.stringify(dados);
        const parametrosJson = JSON.stringify(parametros);
        
        // console.log('📝 [PYTHON SERVICE] Dados JSON:', dadosJson.substring(0, 200) + '...');
        
        // Executar script Python
        console.log('🚀 [PYTHON SERVICE] Executando processo Python...');
        const pythonProcess = spawn(pythonCommand, [
          this.scriptPath,
          '--dados', dadosJson,
          '--parametros', parametrosJson
        ], {
          stdio: ['pipe', 'pipe', 'pipe']
        });
        
        console.log(`🆔 [PYTHON SERVICE] PID do processo: ${pythonProcess.pid}`);

        let stdout = '';
        let stderr = '';

        // Capturar saída
        pythonProcess.stdout.on('data', (data) => {
          const chunk = data.toString();
          stdout += chunk;
          // console.log('📤 [PYTHON SERVICE] STDOUT chunk:', chunk.substring(0, 300));
        });

        pythonProcess.stderr.on('data', (data) => {
          const chunk = data.toString();
          stderr += chunk;
          // console.error('❌ [PYTHON SERVICE] STDERR chunk:', chunk);
        });

        // Tratar conclusão
        pythonProcess.on('close', (code) => {
          console.log(`🏁 [PYTHON SERVICE] Processo finalizado com código: ${code}`);
          if (code !== 0) {
            console.log('📤 [PYTHON SERVICE] STDOUT completo:', stdout);
            console.log('❌ [PYTHON SERVICE] STDERR completo:', stderr);
          }
          
          if (code === 0) {
            try {
              const resultado = JSON.parse(stdout);
              console.log('✅ [PYTHON SERVICE] Algoritmo executado com sucesso!', {
                alocacoes: resultado.alocacoes?.length || 0,
                score: resultado.score_otimizacao
              });
              resolve(resultado);
            } catch (parseError) {
              console.error('💥 [PYTHON SERVICE] Erro ao parsear resultado:', parseError);
              console.error('📝 [PYTHON SERVICE] Conteúdo que falhou ao parsear:', stdout);
              reject(new Error(`Erro ao parsear resultado Python: ${parseError}`));
            }
          } else {
            console.error(`💥 [PYTHON SERVICE] Script falhou com código ${code}`);
            reject(new Error(`Script Python falhou (código ${code}): ${stderr || stdout}`));
          }
        });

        // Tratar erros do processo
        pythonProcess.on('error', (error) => {
          console.error('💥 [PYTHON SERVICE] Erro no processo:', error);
          reject(new Error(`Erro ao executar Python: ${error.message}`));
        });

        // Timeout de segurança (30 segundos)
        setTimeout(() => {
          pythonProcess.kill();
          reject(new Error('Timeout: Script Python demorou mais que 30 segundos'));
        }, 30000);

      } catch (error) {
        reject(new Error(`Erro interno: ${error}`));
      }
    });
  }

  async verificarPython(): Promise<boolean> {
    return new Promise((resolve) => {
      const pythonCommand = process.platform === 'win32' ? 'python' : 'python3';
      
      const pythonProcess = spawn(pythonCommand, ['--version'], { stdio: 'pipe' });
      
      pythonProcess.on('close', (code) => {
        resolve(code === 0);
      });
      
      pythonProcess.on('error', () => {
        resolve(false);
      });
    });
  }

  async testarScript(): Promise<boolean> {
    try {
      console.log('🧪 [PYTHON SERVICE] Iniciando teste do script...');
      
      const dadosTeste = {
        salas: [
          {
            id: 'test-1',
            id_sala: 'sala_test',
            nome: 'Sala Teste',
            capacidade_total: 30,
            localizacao: 'Bloco A - Teste',
            status: 'ATIVA',
            cadeiras_moveis: true,
            cadeiras_especiais: 2
          }
        ],
        turmas: [
          {
            id: 'test-1',
            id_turma: 'turma_test',
            nome: 'Turma Teste',
            alunos: 25,
            duracao_min: 120,
            esp_necessarias: 1
          }
        ]
      };

      const parametrosTeste = {
        priorizar_capacidade: true,
        priorizar_especiais: true,
        priorizar_proximidade: true
      };

      console.log('🔍 [PYTHON SERVICE] Dados de teste preparados:', dadosTeste);
      const resultado = await this.executarAlocacaoInteligente(dadosTeste, parametrosTeste);
      console.log('📊 [PYTHON SERVICE] Resultado do teste:', resultado);
      
      return resultado.success && resultado.alocacoes && resultado.alocacoes.length > 0;
      
    } catch (error) {
      console.error('💥 [PYTHON SERVICE] Erro no teste do script Python:', error);
      return false;
    }
  }
}

export const pythonService = new PythonService();
