const { createServer } = require('http');
const { parse } = require('url');
const next = require('next');
const httpProxy = require('next/dist/compiled/http-proxy');

const dev = process.env.NODE_ENV !== 'production';
const app = next({ dev });
const handle = app.getRequestHandler();

// Create a proxy server for the FastAPI backend
const proxy = httpProxy.createProxyServer({
  target: 'http://127.0.0.1:8000',
  ws: true
});

proxy.on('error', (err, req, res) => {
  console.error('Proxy Error:', err);
  if (res && res.writeHead) {
    res.writeHead(502, { 'Content-Type': 'text/plain' });
    res.end('Proxy Error');
  }
});

app.prepare().then(() => {
  const server = createServer((req, res) => {
    const parsedUrl = parse(req.url, true);
    // Proxy /api/ and /ws/ traffic to the backend
    if (parsedUrl.pathname.startsWith('/api/') || parsedUrl.pathname.startsWith('/ws/')) {
      console.log('Proxying HTTP to backend:', parsedUrl.pathname);
      proxy.web(req, res);
    } else {
      // Let Next.js handle everything else
      handle(req, res, parsedUrl);
    }
  });

  // Intercept WebSocket upgrade requests
  server.on('upgrade', (req, socket, head) => {
    const parsedUrl = parse(req.url, true);
    console.log('Upgrade requested for:', parsedUrl.pathname);
    if (parsedUrl.pathname.startsWith('/ws/')) {
      console.log('Proxying WS to backend...');
      proxy.ws(req, socket, head);
    }
  });

  server.listen(3000, (err) => {
    if (err) throw err;
    console.log('> Custom proxy server ready on http://localhost:3000');
  });
});
