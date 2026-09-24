// Движок в браузере: Pyodide (Python в WebAssembly) с numpy/scipy/scikit-image/Pillow и пакетом legobot.
// Работает в веб-воркере, чтобы страница не замирала на время расчёта.
// Сообщения: {type:"init"} → {type:"progress", text} … {type:"ready"} | {type:"fatal", error}
//            {type:"build", id, image: ArrayBuffer, options} / {type:"regrid", id, codes, options}
//            {type:"model", id, data: ArrayBuffer} — готовая модель .io/.ldr/.mpd
//            → {type:"result", id, files: {имя: Uint8Array|string}, summary} | {type:"error", id, error}

const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";
importScripts(PYODIDE + "pyodide.js");

let pyodide = null, build = null, buildFromGrid = null, buildFromModel = null;
const post = (m) => self.postMessage(m);

async function init(engineVersion) {
  post({ type: "progress", text: "Загружаю Python (≈15 МБ, один раз)…" });
  pyodide = await loadPyodide({ indexURL: PYODIDE });
  post({ type: "progress", text: "Загружаю numpy, scipy, scikit-image…" });
  await pyodide.loadPackage(["numpy", "scipy", "scikit-image", "pillow", "matplotlib"]);
  post({ type: "progress", text: "Загружаю legobot и инструкции…" });
  const response = await fetch(new URL("legobot.zip?v=" + engineVersion, self.location.href));   // относительно worker.js
  if (!response.ok) throw new Error("legobot.zip: HTTP " + response.status);
  const zip = await response.arrayBuffer();
  pyodide.FS.mkdirTree("/app");
  pyodide.unpackArchive(zip, "zip", { extractDir: "/app" });
  await pyodide.runPythonAsync(`
import sys, warnings
sys.path.insert(0, "/app")
warnings.filterwarnings("ignore")
for name in ("trimesh", "rembg", "onnxruntime"):
    sys.modules[name] = None          # этих зависимостей в браузере нет; их импорт должен падать сразу
import matplotlib                     # приходит вместе с scikit-image — на нём инструкция
matplotlib.use("Agg")
import matplotlib.pyplot              # прогрев: первый импорт pyplot в браузере занимает секунды,
from matplotlib.backends.backend_pdf import PdfPages   # пусть это будет на загрузке, а не после кнопки
import legobot.web

def _build(image, opts):              # image — Uint8Array из JS, opts — объект JS
    return legobot.web.build(image.to_bytes(), **opts.to_py())

def _regrid(codes, opts):
    return legobot.web.build_from_grid(codes.to_py(), **opts.to_py())

def _model(data):                     # data — Uint8Array с .io, .ldr или .mpd
    return legobot.web.build_from_model(data.to_bytes())
`);
  build = pyodide.globals.get("_build");
  buildFromGrid = pyodide.globals.get("_regrid");
  buildFromModel = pyodide.globals.get("_model");
  post({ type: "ready" });
}

function toJs(result) {
  // PyProxy словаря → обычные объекты; bytes → Uint8Array (передаём без копирования)
  const out = result.toJs({ dict_converter: Object.fromEntries });
  result.destroy();
  const files = {}, transfer = [];
  for (const [name, value] of Object.entries(out.files)) {
    if (value instanceof Uint8Array) { files[name] = value; transfer.push(value.buffer); }
    else files[name] = value;
  }
  return { files, summary: out.summary, transfer };
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === "init") return await init(m.version);
    if (!build) throw new Error("движок ещё загружается");
    let result;
    if (m.type === "build") result = build(new Uint8Array(m.image), m.options || {});
    else if (m.type === "regrid") result = buildFromGrid(m.codes, m.options || {});
    else if (m.type === "model") result = buildFromModel(new Uint8Array(m.data));
    else return;
    const { files, summary, transfer } = toJs(result);
    self.postMessage({ type: "result", id: m.id, files, summary }, transfer);
  } catch (err) {
    const text = String(err.message || err);
    const known = text.match(/ValueError: (.*)$/m);   // сообщения пайплайна — на русском, показываем как есть
    post({ type: m.type === "init" ? "fatal" : "error", id: m.id, error: known ? known[1] : text.split("\n").slice(-3).join(" ") });
  }
};
