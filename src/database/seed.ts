import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

async function main() {
  console.log('🌱 Iniciando seed do banco de dados...');

  // Criar salas
  const salas = await Promise.all([
    prisma.sala.create({
      data: {
        id_sala: 'sala_1',
        nome: 'Sala 1',
        capacidade_total: 35,
        localizacao: 'Bloco A - 2º andar',
        status: 'ATIVA',
        cadeiras_moveis: true,
        cadeiras_especiais: 2
      }
    }),
    prisma.sala.create({
      data: {
        id_sala: 'sala_2',
        nome: 'Sala 2',
        capacidade_total: 40,
        localizacao: 'Bloco B - 1º andar',
        status: 'ATIVA',
        cadeiras_moveis: false,
        cadeiras_especiais: 0
      }
    }),
    prisma.sala.create({
      data: {
        id_sala: 'sala_3',
        nome: 'Sala 3',
        capacidade_total: 28,
        localizacao: 'Bloco A - 1º andar',
        status: 'ATIVA',
        cadeiras_moveis: true,
        cadeiras_especiais: 1
      }
    }),
    prisma.sala.create({
      data: {
        id_sala: 'sala_4',
        nome: 'Sala 4',
        capacidade_total: 50,
        localizacao: 'Bloco C - 3º andar',
        status: 'ATIVA',
        cadeiras_moveis: true,
        cadeiras_especiais: 3
      }
    })
  ]);

  console.log(`✅ Criadas ${salas.length} salas`);

  // Criar turmas
  const turmas = await Promise.all([
    prisma.turma.create({
      data: {
        id_turma: 'port_101',
        nome: 'Português 101',
        alunos: 32,
        duracao_min: 120,
        esp_necessarias: 1
      }
    }),
    prisma.turma.create({
      data: {
        id_turma: 'mat_101',
        nome: 'Matemática 101',
        alunos: 28,
        duracao_min: 120,
        esp_necessarias: 0
      }
    }),
    prisma.turma.create({
      data: {
        id_turma: 'cien_201',
        nome: 'Ciências 201',
        alunos: 38,
        duracao_min: 120,
        esp_necessarias: 0
      }
    }),
    prisma.turma.create({
      data: {
        id_turma: 'hist_201',
        nome: 'História 201',
        alunos: 45,
        duracao_min: 120,
        esp_necessarias: 2
      }
    })
  ]);

  console.log(`✅ Criadas ${turmas.length} turmas`);

  // Criar projetos
  const projeto1 = await prisma.projetoAlocacao.create({
    data: {
      id_projeto: 'alocacao_matutino',
      nome: 'Alocação Matutino - Bloco A',
      descricao: 'Alocação das turmas do período matutino no Bloco A',
      status: 'CONFIGURACAO'
    }
  });

  const projeto2 = await prisma.projetoAlocacao.create({
    data: {
      id_projeto: 'alocacao_vespertino',
      nome: 'Alocação Vespertino - Bloco B',
      descricao: 'Alocação das turmas do período vespertino no Bloco B',
      status: 'PRONTO'
    }
  });

  console.log(`✅ Criados 2 projetos`);

  // Associar salas e turmas aos projetos
  // Projeto 1 - Matutino
  await Promise.all([
    prisma.projetoSala.create({
      data: {
        projeto_id: projeto1.id,
        sala_id: salas[0].id // Sala 1
      }
    }),
    prisma.projetoSala.create({
      data: {
        projeto_id: projeto1.id,
        sala_id: salas[2].id // Sala 3
      }
    }),
    prisma.projetoTurma.create({
      data: {
        projeto_id: projeto1.id,
        turma_id: turmas[0].id // Português 101
      }
    }),
    prisma.projetoTurma.create({
      data: {
        projeto_id: projeto1.id,
        turma_id: turmas[1].id // Matemática 101
      }
    })
  ]);

  // Projeto 2 - Vespertino
  await Promise.all([
    prisma.projetoSala.create({
      data: {
        projeto_id: projeto2.id,
        sala_id: salas[1].id // Sala 2
      }
    }),
    prisma.projetoSala.create({
      data: {
        projeto_id: projeto2.id,
        sala_id: salas[3].id // Sala 4
      }
    }),
    prisma.projetoTurma.create({
      data: {
        projeto_id: projeto2.id,
        turma_id: turmas[2].id // Ciências 201
      }
    }),
    prisma.projetoTurma.create({
      data: {
        projeto_id: projeto2.id,
        turma_id: turmas[3].id // História 201
      }
    })
  ]);

  console.log(`✅ Associações criadas`);

  console.log('🎉 Seed concluído com sucesso!');
}

main()
  .catch((e) => {
    console.error('❌ Erro durante o seed:', e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
