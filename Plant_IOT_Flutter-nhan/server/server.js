const path = require('path');
const fs = require('fs');
const http = require('http');
const express = require('express');
const { Server } = require('socket.io');
const cors = require('cors');
const helmet = require('helmet');

const { config, validateEnv, buildCorsOriginFunction } = require('./config/env');
const { getDb, runMigrations, cleanOldData, closeDb } = require('./config/database');
const authMiddleware = require('./middleware/auth');
const errorHandler = require('./middleware/errorHandler');
const { requestLogger } = require('./middleware/logger');
const rateLimiter = require('./middleware/rateLimiter');
const sensorRoutes = require('./src/routes/sensors');
const historyRoutes = require('./src/routes/history');
const relayRoutes = require('./src/routes/relay');
const cameraRoutes = require('./src/routes/camera');
const healthRoutes = require('./src/routes/health');

const UPLOADS_DIR = path.resolve(__dirname, config.UPLOADS_DIR);
const LOGS_DIR = path.resolve(__dirname, 'logs');
fs.mkdirSync(UPLOADS_DIR, { recursive: true });
fs.mkdirSync(LOGS_DIR, { recursive: true });

validateEnv();
getDb();
runMigrations();
cleanOldData();

const app = express();
const corsOrigin = buildCorsOriginFunction();

app.set('trust proxy', config.TRUST_PROXY);

const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: config.CORS_ORIGINS ? corsOrigin : true,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    credentials: true,
  },
  allowEIO3: true,
  pingTimeout: 60000,
  pingInterval: 25000,
  connectTimeout: 45000,
});

io.use((socket, next) => {
  const apiKey =
    socket.handshake.headers['x-api-key'] ||
    socket.handshake.auth?.apiKey ||
    socket.handshake.query?.apiKey;
  if (!apiKey || apiKey !== config.API_KEY) {
    const err = new Error('Unauthorized socket: invalid or missing API key');
    err.data = { code: 401 };
    return next(err);
  }
  return next();
});

io.on('connection', (socket) => {
  console.log(`Socket connected: ${socket.id}`);
});

app.locals.io = io;

app.use(helmet({ crossOriginResourcePolicy: { policy: 'cross-origin' } }));
app.use(cors({ origin: corsOrigin, credentials: true }));
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));
app.use(requestLogger);
app.use(rateLimiter);
app.use('/uploads', express.static(UPLOADS_DIR, { maxAge: '1h' }));

app.use('/health', healthRoutes);
app.use('/api', authMiddleware);
app.use('/api/sensors', sensorRoutes);
app.use('/api/sensors', historyRoutes);
app.use('/api/relay', relayRoutes);
app.use('/api/camera', cameraRoutes);

app.use(errorHandler);

const HOST = config.HOST;
const PORT = config.PORT;
const serverInstance = server.listen(PORT, HOST, () => {
  console.log(
    `Plant IoT server listening on http://${HOST}:${PORT} (NODE_ENV=${config.NODE_ENV})`
  );
});

const cleanupTimer = setInterval(() => {
  try {
    cleanOldData();
  } catch (err) {
    console.error('Failed scheduled cleanup:', err);
  }
}, 1000 * 60 * 60 * 12);

function shutdown(signal) {
  console.log(`Received ${signal}, closing server...`);
  clearInterval(cleanupTimer);
  serverInstance.close(() => {
    io.close(() => {
      closeDb();
      console.log('Socket.IO closed');
      process.exit(0);
    });
  });
  setTimeout(() => {
    console.error('Force shutdown after timeout');
    process.exit(1);
  }, 10000).unref();
}

process.on('SIGTERM', () => shutdown('SIGTERM'));
process.on('SIGINT', () => shutdown('SIGINT'));
process.on('uncaughtException', (err) => {
  console.error('Uncaught exception:', err);
  process.exit(1);
});
process.on('unhandledRejection', (reason) => {
  console.error('Unhandled rejection:', reason);
});
