// Экран загрузки: пока движок (Pyodide + legobot) поднимается, страницу с формой показывать
// нечем — кнопка всё равно не работает. Поэтому сначала показываем только его: пиксельную
// сцену, полосу по стадиям и текст того, что сейчас качается.
// Сцена рисуется кодом на канве 96×72 «настоящих» пикселей и растягивается CSS'ом, поэтому
// картинка остаётся пиксельной и ничего не весит.

const BRICK_W = 26, BRICK_H = 7, GROUND = 62, LEFT = 35;
const COLORS = ["#e06c75", "#61afef", "#e5c07b", "#98c379", "#c678dd", "#abb2bf"];
const FALL = 4;              // пикселей за кадр: падение рывками, как в пиксель-арте
const FRAME_MS = 90;

const $ = (id) => document.getElementById(id);
let timer = null, tower = [], falling = null, pause = 0;

function nextBrick() {
  if (tower.length >= 6) { pause = 8; return null; }
  return { y: -BRICK_H, color: COLORS[tower.length % COLORS.length], rest: GROUND - (tower.length + 1) * BRICK_H };
}

function brick(ctx, x, y, color) {
  ctx.fillStyle = color;
  ctx.fillRect(x, y, BRICK_W, BRICK_H);
  ctx.fillStyle = "#16181d";
  ctx.fillRect(x, y + BRICK_H - 1, BRICK_W, 1);          // тень снизу — видно, что деталь объёмная
  ctx.fillStyle = color;
  for (const dx of [4, 12, 20]) ctx.fillRect(x + dx, y - 2, 4, 2);   // штырьки
  ctx.fillStyle = "#16181d";
  for (const dx of [4, 12, 20]) ctx.fillRect(x + dx, y - 2, 4, 1);
}

function frame(ctx) {
  ctx.clearRect(0, 0, 96, 72);
  ctx.fillStyle = "#3e4451";
  ctx.fillRect(LEFT - 6, GROUND, BRICK_W + 12, 2);       // «стол»
  for (const b of tower) brick(ctx, LEFT, b.rest, b.color);
  if (pause > 0) {
    if (--pause === 0) tower = [];
    return;
  }
  if (!falling) { falling = nextBrick(); return; }
  falling.y += FALL;
  if (falling.y >= falling.rest) { tower.push(falling); falling = null; return; }
  brick(ctx, LEFT, falling.y, falling.color);
}

export function start() {
  const canvas = $("loader-art");
  if (!canvas || timer) return;
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  timer = setInterval(() => frame(ctx), FRAME_MS);
}

export function progress(text, share) {
  start();
  $("loader-text").textContent = text;
  if (share != null) $("loader-fill").style.width = Math.round(share * 100) + "%";
}

export function fail(text) {
  clearInterval(timer); timer = null;
  $("loader").classList.add("failed");
  $("loader-text").textContent = text;
}

export function done() {
  clearInterval(timer); timer = null;
  $("loader").hidden = true;
}
