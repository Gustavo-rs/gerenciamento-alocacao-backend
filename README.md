# Gerenciamento de Alocação - Backend

API REST para o sistema de gerenciamento de alocação de salas e turmas.

## 🚀 Tecnologias

- **Node.js** - Runtime JavaScript
- **TypeScript** - Tipagem estática
- **Express.js** - Framework web
- **Prisma** - ORM para banco de dados
- **PostgreSQL** - Banco de dados
- **Zod** - Validação de schemas

## 📋 Pré-requisitos

### Opção 1: Com Docker (Recomendado)
- Docker 20+
- Docker Compose 2+

### Opção 2: Instalação Manual
- Node.js 18+ 
- PostgreSQL 12+
- npm ou yarn

## 🐳 Instalação com Docker (Recomendado)

### Apenas Banco PostgreSQL
```bash
# 1. Clonar e entrar no diretório
git clone <repo>
cd gerenciamento-alocacao-backend

# 2. Iniciar apenas o banco PostgreSQL
npm run docker:db

# 3. Instalar dependências
npm install

# 4. Configurar variáveis de ambiente
cp env.example .env

# 5. Gerar cliente Prisma e aplicar schema
npm run db:generate
npm run db:push

# 6. Popular banco com dados iniciais
npm run db:seed

# 7. Executar backend em modo desenvolvimento
npm run dev
```

### Stack Completa (Banco + Backend)
```bash
# 1. Build e start de todos os serviços
npm run docker:full:build

# 2. Para aplicar schema e seed (primeira vez)
npm run db:push
npm run db:seed
```

### Scripts Docker Úteis
```bash
npm run docker:db          # Apenas PostgreSQL
npm run docker:pgadmin      # PostgreSQL + PgAdmin
npm run docker:up           # Todos os serviços básicos
npm run docker:down         # Parar todos os serviços
npm run docker:logs         # Ver logs dos containers
npm run docker:full         # Stack completa
npm run docker:full:build   # Rebuild da stack completa
```

## 🔧 Instalação Manual

1. **Instalar dependências:**
   ```bash
   npm install
   ```

2. **Configurar PostgreSQL:**
   ```bash
   # Criar banco de dados
   createdb gerenciamento_alocacao
   ```

3. **Configurar variáveis de ambiente:**
   ```bash
   cp env.example .env
   ```
   
   Editar o arquivo `.env` com suas configurações:
   ```env
   DATABASE_URL="postgresql://username:password@localhost:5432/gerenciamento_alocacao"
   PORT=3001
   NODE_ENV=development
   FRONTEND_URL=http://localhost:5173
   ```

4. **Configurar banco de dados:**
   ```bash
   # Gerar cliente Prisma
   npm run db:generate
   
   # Aplicar migrações
   npm run db:push
   
   # Popular banco com dados iniciais
   npm run db:seed
   ```

## 🏃‍♂️ Executando

### Desenvolvimento
```bash
npm run dev
```

### Produção
```bash
npm run build
npm start
```

O servidor estará disponível em `http://localhost:3001`

## 🌐 Acessos dos Serviços

Quando executando com Docker:

- **Backend API**: `http://localhost:3001`
  - Health Check: `http://localhost:3001/health`
  - Documentação da API: Ver seção "API Endpoints" abaixo

- **PostgreSQL**: `localhost:5432`
  - User: `postgres`
  - Password: `postgres123`
  - Database: `gerenciamento_alocacao`

- **PgAdmin** (interface web do PostgreSQL): `http://localhost:8080`
  - Email: `admin@admin.com`
  - Password: `admin123`

## 📚 API Endpoints

### Salas
- `GET /api/salas` - Listar todas as salas
- `GET /api/salas/:id` - Buscar sala por ID
- `POST /api/salas` - Criar nova sala
- `PUT /api/salas/:id` - Atualizar sala
- `DELETE /api/salas/:id` - Deletar sala

### Turmas
- `GET /api/turmas` - Listar todas as turmas
- `GET /api/turmas/:id` - Buscar turma por ID
- `POST /api/turmas` - Criar nova turma
- `PUT /api/turmas/:id` - Atualizar turma
- `DELETE /api/turmas/:id` - Deletar turma

### Projetos
- `GET /api/projetos` - Listar todos os projetos
- `GET /api/projetos/:id` - Buscar projeto por ID
- `POST /api/projetos` - Criar novo projeto
- `PUT /api/projetos/:id` - Atualizar projeto
- `DELETE /api/projetos/:id` - Deletar projeto
- `POST /api/projetos/:id/salas` - Adicionar sala ao projeto
- `DELETE /api/projetos/:id/salas/:salaId` - Remover sala do projeto
- `POST /api/projetos/:id/turmas` - Adicionar turma ao projeto
- `DELETE /api/projetos/:id/turmas/:turmaId` - Remover turma do projeto

### Resultados
- `GET /api/resultados` - Listar todos os resultados
- `GET /api/resultados/:id` - Buscar resultado por ID
- `GET /api/resultados/projeto/:projetoId` - Buscar resultados por projeto
- `POST /api/resultados` - Criar novo resultado
- `POST /api/resultados/executar/:projetoId` - Executar algoritmo de alocação
- `DELETE /api/resultados/:id` - Deletar resultado

### Health Check
- `GET /health` - Status da API

## 📖 Exemplos de Uso

### Criar uma sala
```json
POST /api/salas
{
  "id_sala": "sala_5",
  "nome": "Sala 5",
  "capacidade_total": 30,
  "localizacao": "Bloco A - 3º andar",
  "status": "ATIVA",
  "cadeiras_moveis": 30,
  "cadeiras_especiais": 1
}
```

### Criar uma turma
```json
POST /api/turmas
{
  "id_turma": "eng_301",
  "nome": "Inglês 301",
  "alunos": 25,
  "duracao_min": 90,
  "esp_necessarias": 0
}
```

### Criar um projeto
```json
POST /api/projetos
{
  "id_projeto": "alocacao_noturno",
  "nome": "Alocação Noturno",
  "descricao": "Alocação das turmas do período noturno",
  "status": "CONFIGURACAO"
}
```

### Executar algoritmo de alocação
```json
POST /api/resultados/executar/:projetoId
{
  "priorizar_capacidade": true,
  "priorizar_especiais": true,
  "priorizar_proximidade": false
}
```

## 🗃️ Estrutura do Banco

O banco de dados possui as seguintes tabelas principais:

- **salas** - Informações das salas
- **turmas** - Informações das turmas  
- **projetos_alocacao** - Projetos de alocação
- **projeto_salas** - Relacionamento projeto-salas
- **projeto_turmas** - Relacionamento projeto-turmas
- **resultados_alocacao** - Resultados das alocações
- **alocacoes** - Alocações específicas sala-turma

## 🔒 Validação

Todas as entradas são validadas usando Zod schemas. Campos obrigatórios e tipos são verificados automaticamente.

## 🐛 Debug

Para debug detalhado, configure:
```env
NODE_ENV=development
```

Os logs incluem:
- Requisições HTTP (Morgan)
- Erros de validação
- Erros de banco de dados
- Stack traces completos

## 📝 Scripts Disponíveis

- `npm run dev` - Servidor de desenvolvimento com hot reload
- `npm run build` - Build para produção
- `npm start` - Executar versão de produção
- `npm run db:generate` - Gerar cliente Prisma
- `npm run db:push` - Aplicar schema ao banco
- `npm run db:migrate` - Criar e aplicar migrações
- `npm run db:seed` - Popular banco com dados iniciais
