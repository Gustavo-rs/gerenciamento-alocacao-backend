import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import { PrismaClient } from '@prisma/client';

// Rotas
import salasRoutes from './routes/salas';
import turmasRoutes from './routes/turmas';
import projetosRoutes from './routes/projetos';
import resultadosRoutes from './routes/resultados';
import alocacoesRoutes from './routes/alocacoes';
import horariosRoutes from './routes/horarios';

const app = express();
const prisma = new PrismaClient();

// Middleware
app.use(helmet());
app.use(morgan('combined'));
app.use(cors({
  origin: process.env.FRONTEND_URL || 'http://localhost:5173',
  credentials: true
}));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Middleware para disponibilizar prisma nas requisições
app.use((req, res, next) => {
  req.prisma = prisma;
  next();
});

// Rotas
app.use('/api/salas', salasRoutes);
app.use('/api/turmas', turmasRoutes);
app.use('/api/projetos', projetosRoutes);
app.use('/api/resultados', resultadosRoutes);
app.use('/api/alocacoes', alocacoesRoutes);
app.use('/api/horarios', horariosRoutes);

// Rota de health check
app.get('/health', (req, res) => {
  res.json({ 
    status: 'OK', 
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  });
});

// Middleware de erro
app.use((err: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
  console.error(err.stack);
  res.status(500).json({
    error: 'Algo deu errado!',
    message: process.env.NODE_ENV === 'development' ? err.message : 'Erro interno do servidor'
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({ error: 'Rota não encontrada' });
});

const PORT = process.env.PORT || 3001;

// Graceful shutdown
process.on('SIGINT', async () => {
  console.log('\n🔄 Fechando servidor...');
  await prisma.$disconnect();
  process.exit(0);
});

process.on('SIGTERM', async () => {
  console.log('\n🔄 Fechando servidor...');
  await prisma.$disconnect();
  process.exit(0);
});

app.listen(PORT, () => {
  console.log(`🚀 Servidor rodando em http://localhost:${PORT}`);
  console.log(`📊 Health check disponível em http://localhost:${PORT}/health`);
});

export default app;
