// Страница legobot: загрузка фото -> опрос задачи -> вьюшка модели (three.js LDrawLoader),
// сводка, самопроверка, список деталей, скачивание. Без бэкенда показывает демо из /demo.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { LDrawLoader } from "three/addons/loaders/LDrawLoader.js";
import { LDrawConditionalLineMaterial } from "three/addons/materials/LDrawConditionalLineMaterial.js";

const API = new URLSearchParams(location.search).get("api") || window.LEGOBOT_API || "";
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
const colorTable = fetch("catalog/colors.csv").then((r) => r.text()).then((text) =>
  Object.fromEntries(text.trim().split("\n").slice(1).map((l) => l.split(",")).map((c) => [c[1], c[3]])));
async function colorHex(name) {
  return "#" + ((await colorTable)[name] || "888888");
}

async function showResult(files, summary, title) {
  $("result").hidden = false;
  $("title").textContent = title;
  const [w, h] = summary.pixels, [sx, sy, sz] = summary.size_cm;
  $("summary").innerHTML = [
    ["Пикселей", `${w} × ${h}`], ["Деталей", summary.parts], ["Шагов", summary.steps],
    ["Размер", `${sx} × ${sy} × ${sz} см`],
  ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
  $("accuracy").textContent = summary.accuracy;
  $("check").src = files["check.png"];
  $("downloads").innerHTML = [
    ["model.io", "Открыть в Studio (.io)"], ["instructions.pdf", "Инструкция (PDF)"], ["parts.csv", "Список деталей (CSV)"], ["model.mpd", "Модель (LDraw)"],
  ].map(([f, label]) => `<a href="${files[f]}" download>${label}</a>`).join("");
  const rows = await Promise.all(summary.colors.map(async ([name, n]) =>
    `<tr><td>${n}</td><td><i class="swatch" style="background:${await colorHex(name)}"></i>${name.replaceAll("_", " ")}</td></tr>`));
  $("parts").innerHTML = rows.join("");
  await showModel(files["model.mpd"]);
}

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
  $("submit").disabled = !API;
  if (!API) $("status").textContent = "Бэкенд не подключён: добавь ?api=адрес к ссылке.";
}

$("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!file || !API) return;
  $("submit").disabled = true;
  const body = new FormData();
  body.append("photo", file);
  $("status").textContent = "Загружаю…";
  try {
    const job = await (await fetch(`${API}/jobs`, { method: "POST", body })).json();
    const started = Date.now();
    for (;;) {
      await new Promise((r) => setTimeout(r, 3000));
      const state = await (await fetch(`${API}/jobs/${job.id}`)).json();
      $("status").textContent = `${state.status === "queued" ? "В очереди" : "Считаю"}… ${Math.round((Date.now() - started) / 1000)} с`;
      if (state.status === "done") {
        $("status").textContent = "Готово";
        const files = Object.fromEntries(Object.entries(state.files).map(([k, v]) => [k, API + v]));
        await showResult(files, state.summary, file.name);
        break;
      }
      if (state.status === "error") { $("status").textContent = "Ошибка: " + state.error; break; }
    }
  } catch (err) {
    $("status").textContent = "Не удалось связаться с сервером: " + err.message;
  }
  $("submit").disabled = false;
});

// --- демо ---
(async () => {
  const summary = await (await fetch("demo/summary.json")).json();
  const files = Object.fromEntries(["model.mpd", "model.io", "instructions.pdf", "parts.csv", "check.png"].map((f) => [f, `demo/${f}`]));
  await showResult(files, summary, "Пример: Марио (Pixel Pals)");
})();
