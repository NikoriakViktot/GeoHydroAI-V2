# AGENTS.md — GeoHydroAI-V2

Правила для AI-агентів (Claude Code та інших), що працюють із цим репозиторієм або з сервером `geoai`.

## nginx — одне джерело правди

- Маршрутизацію `geohydroai.org` описує **тільки** `services/nginx/nginx.conf` у цьому репозиторії.
  Повні правила: [`services/nginx/README.md`](services/nginx/README.md). Прочитайте їх перед будь-якою правкою nginx.
- Не створюйте сніпетів nginx в інших репозиторіях і не пишіть інструкцій «вставте цей блок у nginx.conf».
  Якщо сервісу потрібен маршрут — PR у цей файл.
- Upstream до контейнера — лише динамічний (`set $x http://name:port` або `server name:port resolve` у `upstream`
  із `zone`). Статичний `server name:port;` дає 502 після перестворення контейнера.
- Перед PR: `services/nginx/tests/check_config.sh`. Після деплою, що змінив `nginx.conf`: `docker restart geoai-nginx`
  (не `reload` — після `git reset` змонтовано старий inode) і `services/nginx/tests/smoke.sh`. Новий маршрут → новий рядок у `smoke.sh`.
- Резервні копії конфігу — тільки `~/nginx-backups/` на сервері.

## Деплой (небезпечно)

- Пуш у `main` = автодеплой на сервер: `git reset --hard origin/main` у `~/geoai` + `docker compose up -d`.
  **Незакомічені зміни в `~/geoai` буде стерто.** Перед злиттям у `main` перевірте `git -C ~/geoai status`;
  якщо там є зміни — спершу закомітьте їх у гілку й повідомте людину.
- Не зливайте PR у `main` і не пушіть у `main` без явного дозволу людини.
- Правки на живому сервері (nginx reload, перезапуск контейнерів) — лише з дозволу людини; кожну таку правку того ж
  дня треба закомітити в цей репозиторій.

## Секрети

- Паролі/ключі — у `~/geoai/.env`, `~/hydro-secrets/`, `~/geoai-django-secrets/` на сервері. Не виводьте їх у чат,
  логи чи коміти; передавайте в команди через змінні оточення.
