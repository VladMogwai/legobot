// Десктоп legobot: запускает локальный бэкенд (PyInstaller-сборка service/run_server.py)
// и показывает его страницу в окне. Всё на этой машине: без туннелей, Studio открывается кнопкой.
const { app, BrowserWindow, dialog, shell } = require("electron");
const { spawn } = require("child_process");
const net = require("net");
const path = require("path");
const fs = require("fs");
const http = require("http");

let backend = null;

function backendCommand(port) {
  const packaged = path.join(process.resourcesPath, "backend", "legobot-server");
  if (app.isPackaged || fs.existsSync(packaged)) return { cmd: packaged, args: [String(port)], cwd: path.dirname(packaged) };
  const root = path.resolve(__dirname, "..");
  const built = path.join(root, "dist", "legobot-server", "legobot-server");
  if (fs.existsSync(built)) return { cmd: built, args: [String(port)], cwd: path.dirname(built) };
  return { cmd: path.join(root, ".venv", "bin", "python"), args: ["service/run_server.py", String(port)], cwd: root, env: { PYTHONPATH: root } };
}

function freePort() {
  return new Promise((resolve) => {
    const s = net.createServer();
    s.listen(0, "127.0.0.1", () => { const p = s.address().port; s.close(() => resolve(p)); });
  });
}

function waitHealthy(port, timeoutMs) {
  const started = Date.now();
  return new Promise((resolve, reject) => {
    const tick = () => {
      http.get(`http://127.0.0.1:${port}/health`, (res) => (res.statusCode === 200 ? resolve() : retry())).on("error", retry);
    };
    const retry = () => (Date.now() - started > timeoutMs ? reject(new Error("бэкенд не запустился за " + timeoutMs / 1000 + " с")) : setTimeout(tick, 500));
    tick();
  });
}

async function start() {
  const port = await freePort();
  const { cmd, args, cwd, env } = backendCommand(port);
  backend = spawn(cmd, args, { cwd, env: { ...process.env, ...(env || {}) }, stdio: ["ignore", "pipe", "pipe"] });
  backend.stdout.on("data", (d) => process.stdout.write(d));
  backend.stderr.on("data", (d) => process.stderr.write(d));
  backend.on("exit", (code) => { if (!app.isQuitting) { dialog.showErrorBox("legobot", "Бэкенд остановился (код " + code + ")"); app.quit(); } });

  const win = new BrowserWindow({ width: 1280, height: 900, title: "legobot", show: false, webPreferences: { contextIsolation: true } });
  win.webContents.setWindowOpenHandler(({ url }) => { shell.openExternal(url); return { action: "deny" }; });
  win.loadURL("data:text/html;charset=utf-8," + encodeURIComponent("<body style='font:16px system-ui;padding:40px;color:#444'>Запускаю legobot… первый старт занимает до 20 секунд.</body>"));
  win.once("ready-to-show", () => win.show());
  try {
    await waitHealthy(port, 90000);
    await win.loadURL(`http://127.0.0.1:${port}/`);
  } catch (e) {
    dialog.showErrorBox("legobot", e.message);
    app.quit();
  }
}

app.whenReady().then(start);
app.on("before-quit", () => { app.isQuitting = true; if (backend) backend.kill(); });
app.on("window-all-closed", () => app.quit());
