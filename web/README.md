# legobot web

Статическая страница: загрузка фото, вьюшка модели (three.js LDrawLoader), инструкция, детали.
Без сборки. Демо-результат лежит в `public/demo/`, каталог деталей — в `public/catalog/`.

Деплой на Vercel: Root Directory = `web/public`, Framework = Other. Адрес бэкенда — в `public/config.js`.
Локально: `python -m http.server 8765 --directory web/public` и `?api=http://127.0.0.1:7860`.
