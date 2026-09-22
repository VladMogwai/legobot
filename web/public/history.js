// История моделей: всё, что собрано на этой странице, лежит в IndexedDB браузера —
// файлы модели, сводка и сетка пикселей. Сервера нет, значит история живёт на этом устройстве
// и в этом браузере: её видит только владелец, и чистка данных сайта её стирает.

const DB_NAME = "legobot";
const STORE = "models";
const LIMIT = 60;          // сколько моделей храним; самые старые вытесняются

function open() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE, { keyPath: "id" }).createIndex("saved", "saved");
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function run(mode, fn) {
  return open().then((db) => new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, mode);
    const result = fn(tx.objectStore(STORE));
    tx.oncomplete = () => resolve(result?.result ?? result);
    tx.onerror = () => reject(tx.error);
  }));
}

/** Сохранить модель. files — {имя: Uint8Array | string}, thumb — Blob превью. */
export async function save({ title, summary, files, thumb }) {
  const record = {
    id: String(Date.now()) + Math.random().toString(36).slice(2, 6),
    saved: Date.now(), title, summary,
    files: Object.fromEntries(Object.entries(files).map(([name, data]) => [name, blobFor(name, data)])),
    thumb,
  };
  await run("readwrite", (store) => store.put(record));
  await prune();
  return record;
}

export function list() {
  return run("readonly", (store) => store.getAll()).then((rows) => rows.sort((a, b) => b.saved - a.saved));
}

export function get(id) {
  return run("readonly", (store) => store.get(id));
}

export function remove(id) {
  return run("readwrite", (store) => store.delete(id));
}

export async function clear() {
  await run("readwrite", (store) => store.clear());
}

/** Сколько места занято, в мегабайтах (оценка браузера по всему сайту). */
export async function usageMb() {
  const estimate = await navigator.storage?.estimate?.();
  return estimate?.usage ? estimate.usage / 1048576 : null;
}

async function prune() {
  const rows = await list();
  for (const row of rows.slice(LIMIT)) await remove(row.id);
}

export function blobFor(name, data) {
  const type = name.endsWith(".png") ? "image/png"
    : name.endsWith(".csv") ? "text/csv"
    : name.endsWith(".io") ? "application/zip"
    : name.endsWith(".pdf") ? "application/pdf" : "text/plain";
  return data instanceof Blob ? data : new Blob([data], { type });
}
