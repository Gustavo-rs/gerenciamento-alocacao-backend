# Multi-stage build para otimizar o tamanho da imagem
FROM node:18-alpine AS builder

# Definir diretório de trabalho
WORKDIR /app

# Copiar arquivos de dependências
COPY package*.json ./
COPY tsconfig.json ./

# Instalar dependências
RUN npm ci --only=production && npm cache clean --force

# Copiar código fonte
COPY src/ ./src/
COPY prisma/ ./prisma/

# Gerar cliente Prisma
RUN npx prisma generate

# Build da aplicação
RUN npm run build

# Estágio de produção
FROM node:18-alpine AS production

# Criar usuário não-root para segurança
RUN addgroup -g 1001 -S nodejs
RUN adduser -S backend -u 1001

# Definir diretório de trabalho
WORKDIR /app

# Copiar arquivos necessários do builder
COPY --from=builder --chown=backend:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=backend:nodejs /app/dist ./dist
COPY --from=builder --chown=backend:nodejs /app/package*.json ./
COPY --from=builder --chown=backend:nodejs /app/prisma ./prisma

# Instalar apenas dependências de produção
RUN npm ci --only=production && npm cache clean --force

# Mudar para usuário não-root
USER backend

# Expor porta
EXPOSE 3001

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node -e "const http=require('http');const options={hostname:'localhost',port:3001,path:'/health',timeout:2000};const req=http.request(options,(res)=>{process.exit(res.statusCode===200?0:1);});req.on('error',()=>process.exit(1));req.end();"

# Comando para iniciar a aplicação
CMD ["npm", "start"]
