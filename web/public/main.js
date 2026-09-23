// Страница legobot: загрузка фото -> опрос задачи -> вьюшка модели (three.js LDrawLoader),
// сводка, самопроверка, список деталей, скачивание. Без бэкенда показывает демо из /demo.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { LDrawLoader } from "three/addons/loaders/LDrawLoader.js";
import { LDrawConditionalLineMaterial } from "three/addons/materials/LDrawConditionalLineMaterial.js";
import * as history from "./history.js";

const API = new URLSearchParams(location.search).get("api") || window.LEGOBOT_API || "";
// Без бэкенда модель считается прямо в браузере: Pyodide + legobot в веб-воркере (engine/worker.js).
const engine = { worker: null, ready: false, pending: new Map(), seq: 0 };
function startEngine() {
  const el = $("health");
  el.textContent = "● движок загружается…"; el.className = "health";
  engine.worker = new Worker("engine/worker.js?v=" + (window.LEGOBOT_ENGINE || ""));   // версия — чтобы браузер не взял старый воркер из кэша
  engine.worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === "progress") { el.textContent = "● " + m.text; $("status").textContent = m.text + " Кнопка «Собрать» оживёт, когда движок будет готов."; }
    else if (m.type === "ready") { engine.ready = true; online = true; el.textContent = "● считает в браузере"; el.className = "health on"; if (file) $("submit").disabled = false; $("status").textContent = file ? "Готов собирать" : ""; }
    else if (m.type === "fatal") { el.textContent = "● движок не загрузился: " + m.error; el.className = "health off"; $("status").textContent = "Движок не загрузился: " + m.error; }
    else if (m.id && engine.pending.has(m.id)) { engine.pending.get(m.id)(m); engine.pending.delete(m.id); }
  };
  engine.worker.onerror = (e) => { el.textContent = "● движок не загрузился: " + e.message; el.className = "health off"; };
  engine.worker.postMessage({ type: "init", version: window.LEGOBOT_ENGINE || "" });
}
function engineCall(message, transfer) {
  return new Promise((resolve) => {
    const id = ++engine.seq;
    engine.pending.set(id, resolve);
    engine.worker.postMessage({ ...message, id }, transfer || []);
  });
}
async function runLocal(message, transfer, statusEl, title) {
  const started = Date.now();
  const tick = setInterval(() => { statusEl.textContent = `Считаю в браузере… ${Math.round((Date.now() - started) / 1000)} с`; }, 500);
  try {
    const m = await engineCall(message, transfer);
    clearInterval(tick);
    if (m.type === "error") { statusEl.textContent = "Ошибка: " + m.error; return; }
    statusEl.textContent = `Готово за ${Math.round((Date.now() - started) / 1000)} с`;
    const blobs = Object.fromEntries(Object.entries(m.files).map(([name, data]) => [name, history.blobFor(name, data)]));
    await showFiles(blobs, m.summary, title);
    await saveToHistory(title, m.summary, blobs);
  } catch (err) {
    clearInterval(tick);
    statusEl.textContent = "Ошибка: " + err.message;
  }
}
function buildOptions() {
  return {
    mode: $("mode").value, background: "auto",   // фон решают тип модели и сама картинка
    width: +($("width").value || 0), contrast: $("contrast").checked, base: $("base").checked,
  };
}
const $ = (id) => document.getElementById(id);

// --- вьюшка ---
const canvas = $("viewer");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf6f5f2);
const camera = new THREE.PerspectiveCamera(35, 1, 1, 100000);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xffffff, 0x777777, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.2);
sun.position.set(-1, 2, 1.5);
scene.add(sun);
let model = null;
let steps = 1;

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w * renderer.getPixelRatio() || canvas.height !== h * renderer.getPixelRatio()) {
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
}
function animate() {
  resize();
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}
animate();

async function showModel(url) {
  if (model) scene.remove(model);
  const loader = new LDrawLoader();
  loader.setConditionalLineMaterial(LDrawConditionalLineMaterial);
  loader.smoothNormals = false;
  model = await loader.loadAsync(url);
  model.rotation.x = Math.PI;                       // в LDraw ось Y смотрит вниз
  scene.add(model);
  steps = model.userData.numBuildingSteps || 1;
  $("step").max = steps;
  $("step").value = steps;
  applyStep(steps);
  const box = new THREE.Box3().setFromObject(model);
  const size = box.getSize(new THREE.Vector3()), center = box.getCenter(new THREE.Vector3());
  const radius = size.length() / 2;
  camera.position.copy(center).add(new THREE.Vector3(-1.2, 0.9, 1.6).normalize().multiplyScalar(radius * 2.6));
  camera.near = radius / 50; camera.far = radius * 20;
  camera.updateProjectionMatrix();
  controls.target.copy(center);
}

function applyStep(n) {
  model?.traverse((o) => {
    if (o.userData.buildingStep !== undefined) o.visible = o.userData.buildingStep < n;
  });
  $("step-label").textContent = `шаг ${n} из ${steps}`;
}
$("step").addEventListener("input", (e) => applyStep(+e.target.value));

// --- результат ---
const colors = fetch("catalog/colors.csv").then((r) => r.text()).then((text) =>
  text.trim().split(/\r?\n/).slice(1).map((l) => l.split(",")).map(([code, name, , studio, solid, common]) =>
    ({ code: +code, name, hex: "#" + studio, common: solid === "1" && common === "1" })));
async function colorHex(name) {
  return (await colors).find((c) => c.name === name)?.hex || "#888";
}

async function showResult(files, summary, title) {
  $("result").hidden = false;
  $("title").textContent = title;
  const [sx, sy, sz] = summary.size_cm;
  $("summary").innerHTML = [
    summary.kind && ["Тип", summary.kind], summary.pixels && ["Пикселей", `${summary.pixels[0]} × ${summary.pixels[1]}`],
    ["Деталей", summary.parts], ["Шагов", summary.steps],
    summary.pages && ["Инструкция", `${summary.pages} ${plural(summary.pages, "страница", "страницы", "страниц")}`],
    ["Размер", `${sx} × ${sy} × ${sz} см`],
    summary.skipped && ["Пропущено деталей", summary.skipped],
  ].filter(Boolean).map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
  $("accuracy").textContent = summary.accuracy || "";
  $("check").hidden = !files["check.png"];
  if (files["check.png"]) $("check").src = files["check.png"];
  $("editor").hidden = !summary.grid;
  $("downloads").innerHTML = [
    ["model.io", "Скачать .io", "model.io"], ["instructions.pdf", "Инструкция (PDF)", "instructions.pdf"], ["chart.png", "Схема панно (PNG)", "chart.png"],
    ["parts.csv", "Список деталей (CSV)", "parts.csv"], ["model.mpd", "Модель (LDraw)", "model.mpd"],
  ].filter(([f]) => files[f]).map(([f, label, name]) => `<a href="${files[f]}" download="${name}">${label}</a>`).join("");
  if (summary.price_usd != null) $("summary").insertAdjacentHTML("beforeend", `<dt>Pick a Brick</dt><dd>$${summary.price_usd}</dd>`);
  const jobId = files["model.io"].match(/\/jobs\/([^/]+)\//)?.[1];
  if (local && jobId) {
    const b = document.createElement("a");
    b.href = "#"; b.textContent = "Открыть в Studio"; b.className = "primary";
    b.onclick = async (e) => { e.preventDefault(); const r = await fetch(`${API}/jobs/${jobId}/open`, { method: "POST" }); b.textContent = r.ok ? "Открыто в Studio" : "Studio не найден"; };
    $("downloads").prepend(b);
  }
  const rows = await Promise.all(summary.colors.map(async ([name, n]) =>
    `<tr><td>${n}</td><td><i class="swatch" style="background:${await colorHex(name)}"></i>${name.replaceAll("_", " ")}</td></tr>`));
  $("parts").innerHTML = rows.join("");
  await showModel(files["model.mpd"]);
}

// --- история моделей (IndexedDB, только на этом устройстве) ---
async function showFiles(blobs, summary, title) {
  const urls = Object.fromEntries(Object.entries(blobs).map(([name, blob]) => [name, URL.createObjectURL(blob)]));
  await showResult(urls, summary, title);
  if (summary.grid) await showEditor(summary.grid);
}

async function saveToHistory(title, summary, blobs) {
  try {
    await history.save({ title, summary, files: blobs, thumb: blobs["check.png"] || blobs["chart.png"] });
    await renderHistory();
  } catch (err) {
    $("history-note").textContent = "История не сохранилась: " + err.message;
  }
}

function historyCard(row) {
  const date = new Date(row.saved).toLocaleString("ru-RU", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
  const price = row.summary.price_usd != null ? ` · $${row.summary.price_usd}` : "";
  const thumb = row.thumb ? `<img src="${URL.createObjectURL(row.thumb)}" alt="">` : "";
  return `<article class="saved" data-id="${row.id}">
      ${thumb}
      <div class="saved-meta">
        <b>${row.title}</b>
        <span>${row.summary.kind} · ${row.summary.parts} дет.${price}</span>
        <span class="saved-date">${date}</span>
        <div class="saved-actions">
          <button type="button" data-act="open">Открыть</button>
          <span data-files></span>
          <button type="button" data-act="remove" class="link-danger">Удалить</button>
        </div>
      </div>
    </article>`;
}

async function renderHistory() {
  const rows = await history.list();
  $("history").hidden = rows.length === 0;
  $("history-list").innerHTML = rows.map(historyCard).join("");
  for (const [id, blobs] of rows.map((r) => [r.id, r.files])) {
    const holder = $("history-list").querySelector(`[data-id="${id}"] [data-files]`);
    holder.innerHTML = Object.keys(blobs)
      .filter((name) => name !== "check.png" && name !== "model.mpd")   // превью и файл вьюшки не для скачивания
      .map((name) => `<a href="${URL.createObjectURL(blobs[name])}" download="${name}">${name}</a>`).join("");
  }
  const mb = await history.usageMb();
  $("history-note").textContent = `${rows.length} ${plural(rows.length, "модель", "модели", "моделей")} в этом браузере` + (mb ? `, ${mb.toFixed(1)} МБ` : "");
}

function plural(n, one, few, many) {
  const mod10 = n % 10, mod100 = n % 100;
  if (mod10 === 1 && mod100 !== 11) return one;
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 10 || mod100 >= 20)) return few;
  return many;
}

$("history-list").addEventListener("click", async (e) => {
  const button = e.target.closest("button");
  if (!button) return;
  const id = button.closest("[data-id]").dataset.id;
  if (button.dataset.act === "remove") {
    await history.remove(id);
    await renderHistory();
    return;
  }
  const row = await history.get(id);
  await showFiles(row.files, row.summary, row.title);
  $("result").scrollIntoView({ behavior: "smooth", block: "start" });
});

$("history-clear").addEventListener("click", async () => {
  if (!confirm("Удалить все сохранённые модели из этого браузера?")) return;
  await history.clear();
  await renderHistory();
});

renderHistory();

// --- жив ли сервис (бэкенд крутится на домашнем Mac) ---
let online = false, local = false;
async function checkHealth() {
  const el = $("health");
  if (!API) { if (!engine.worker) startEngine(); return; }   // без бэкенда — движок в браузере, один на страницу
  try {
    const ctrl = new AbortController();
    setTimeout(() => ctrl.abort(), 8000);
    const r = await fetch(`${API}/health`, { signal: ctrl.signal });
    online = r.ok;
    if (online) local = !!(await r.json()).local;
  } catch { online = false; }
  el.textContent = online ? "● сервис онлайн" : "● сервис офлайн — Mac выключен или не запущен";
  el.className = "health " + (online ? "on" : "off");
  $("submit").disabled = !file || !online;
}
checkHealth();
setInterval(checkHealth, 30000);

// --- загрузка и опрос ---
const photo = $("photo"), drop = $("drop");
photo.addEventListener("change", () => pick(photo.files[0]));
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); pick(e.dataTransfer.files[0]); });
let file = null;
function pick(f) {
  if (!f) return;
  file = f;
  $("preview").src = URL.createObjectURL(f);
  $("preview").hidden = false;
  $("drop-text").textContent = f.name;
  $("submit").disabled = !online;
  if (!online) $("status").textContent = API ? "Сервис сейчас офлайн — попробуй позже." : "Движок ещё загружается — кнопка оживёт через несколько секунд.";
}

async function runJob(request, statusEl, title) {
  statusEl.textContent = "Загружаю…";
  try {
    const job = await (await fetch(`${API}${request.path}`, request.init)).json();
    if (job.detail) throw new Error(job.detail);
    const started = Date.now();
    for (;;) {
      await new Promise((r) => setTimeout(r, 1500));
      const state = await (await fetch(`${API}/jobs/${job.id}`)).json();
      statusEl.textContent = `${state.status === "queued" ? "В очереди" : state.summary?.stage || "Считаю"}… ${Math.round((Date.now() - started) / 1000)} с`;
      if (state.status === "done") {
        statusEl.textContent = "Готово";
        const files = Object.fromEntries(Object.entries(state.files).map(([k, v]) => [k, API + v]));
        await showResult(files, state.summary, title);
        if (state.summary.grid) await showEditor(state.summary.grid);
        return;
      }
      if (state.status === "error") { statusEl.textContent = "Ошибка: " + state.error; return; }
    }
  } catch (err) {
    statusEl.textContent = "Ошибка: " + err.message;
  }
}

$("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!file) return;
  $("submit").disabled = true;
  if (!API) {
    const image = await file.arrayBuffer();
    await runLocal({ type: "build", image, options: buildOptions() }, [image], $("status"), file.name);
    $("submit").disabled = false;
    return;
  }
  const body = new FormData();
  body.append("photo", file);
  body.append("width", $("width").value || "0");
  body.append("background", $("mode").value === "flat" ? "keep" : "cut");
  body.append("contrast", $("contrast").checked ? "true" : "false");
  await runJob({ path: "/jobs", init: { method: "POST", body } }, $("status"), file.name);
  $("submit").disabled = false;
});

$("model-file").addEventListener("change", async () => {
  const f = $("model-file").files[0];
  if (!f) return;
  if (!API) { $("status").textContent = "Загрузка своих .io работает только с бэкендом."; return; }
  const body = new FormData();
  body.append("model", f);
  await runJob({ path: "/models", init: { method: "POST", body } }, $("status"), f.name);
});

// --- редактор пикселей ---
const CELL = 18;
let grid = null, undoStack = [], brush = 0, painting = false;
const gridCanvas = $("grid"), gctx = gridCanvas.getContext("2d");
let hexByCode = {};

async function showEditor(codes) {
  grid = codes.map((col) => col.slice());
  undoStack = [];
  const all = await colors;
  hexByCode = Object.fromEntries(all.map((c) => [c.code, c.hex]));
  const used = new Set(grid.flat().filter((c) => c >= 0));
  const palette = all.filter((c) => c.common || used.has(c.code));
  $("palette").innerHTML = `<button class="erase" data-code="-1" title="Стереть"></button>` +
    palette.map((c) => `<button data-code="${c.code}" title="${c.name.replaceAll("_", " ")}" style="--sw:${c.hex}"></button>`).join("");
  selectBrush(palette[0].code);
  $("editor").hidden = false;
  drawGrid();
}
function selectBrush(code) {
  brush = code;
  for (const b of $("palette").children) b.classList.toggle("selected", +b.dataset.code === code);
}
$("palette").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) selectBrush(+b.dataset.code); });

function drawGrid() {
  const w = grid.length, h = grid[0].length;
  gridCanvas.width = w * CELL + 1; gridCanvas.height = h * CELL + 1;
  gctx.fillStyle = "#ececec"; gctx.fillRect(0, 0, gridCanvas.width, gridCanvas.height);
  for (let x = 0; x < w; x++) for (let y = 0; y < h; y++) {
    gctx.fillStyle = grid[x][y] >= 0 ? hexByCode[grid[x][y]] || "#888" : "#fff";
    gctx.fillRect(x * CELL + 1, y * CELL + 1, CELL - 1, CELL - 1);
  }
}
function cellAt(e) {
  const r = gridCanvas.getBoundingClientRect();
  const x = Math.floor((e.clientX - r.left) / CELL), y = Math.floor((e.clientY - r.top) / CELL);
  return x >= 0 && y >= 0 && x < grid.length && y < grid[0].length ? [x, y] : null;
}
function paint(e) {
  const c = cellAt(e);
  if (!c) return;
  const [x, y] = c, code = e.buttons === 2 || e.button === 2 ? -1 : brush;
  if (grid[x][y] === code) return;
  undoStack.push([x, y, grid[x][y]]);
  grid[x][y] = code;
  gctx.fillStyle = code >= 0 ? hexByCode[code] : "#fff";
  gctx.fillRect(x * CELL + 1, y * CELL + 1, CELL - 1, CELL - 1);
}
gridCanvas.addEventListener("contextmenu", (e) => e.preventDefault());
gridCanvas.addEventListener("pointerdown", (e) => { painting = true; paint(e); });
gridCanvas.addEventListener("pointermove", (e) => { if (painting) paint(e); });
window.addEventListener("pointerup", () => { painting = false; });
$("undo").addEventListener("click", () => {
  const last = undoStack.pop();
  if (!last) return;
  const [x, y, code] = last;
  grid[x][y] = code;
  drawGrid();
});
$("rebuild").addEventListener("click", async () => {
  $("rebuild").disabled = true;
  if (!API) {
    await runLocal({ type: "regrid", codes: grid, options: { mode: $("mode").value, base: $("base").checked } }, [], $("editor-status"), $("title").textContent + " (правка)");
    $("rebuild").disabled = false;
    return;
  }
  await runJob({ path: "/grids", init: { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ codes: grid }) } },
    $("editor-status"), $("title").textContent + " (правка)");
  $("rebuild").disabled = false;
});

// --- демо ---
(async () => {
  const summary = await (await fetch("demo/summary.json")).json();
  const files = Object.fromEntries(["model.mpd", "model.io", "instructions.pdf", "parts.csv", "check.png"].map((f) => [f, `demo/${f}`]));
  await showResult(files, summary, "Пример: Марио (Pixel Pals)");
  if (summary.grid) await showEditor(summary.grid);
})();
