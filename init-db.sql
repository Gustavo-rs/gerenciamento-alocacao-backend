-- Script de inicialização do banco de dados
-- Este arquivo é executado automaticamente quando o container PostgreSQL é criado

-- Criar extensões úteis
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Configurações de encoding
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;

-- Mensagem de confirmação
SELECT 'Banco de dados gerenciamento_alocacao inicializado com sucesso!' as status;
