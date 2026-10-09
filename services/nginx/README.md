# nginx на geohydroai.org — одне джерело правди

**Правило:** маршрутизацію `geohydroai.org` (і `python-course-viktor-nikoriak.org`) описує **лише один файл** —
[`services/nginx/nginx.conf`](nginx.conf) у репозиторії **GeoHydroAI-V2**. Інших копій, сніпетів чи «вставок» у
інших репозиторіях немає. Те, що працює на сервері, = те, що лежить у `main` цього файлу.

## 1. Як це влаштовано

| Що | Де |
|---|---|
| Єдиний nginx | контейнер `geoai-nginx` (`nginx:stable`, порти 80/443), compose-проєкт `~/geoai` (GeoHydroAI-V2) |
| Конфіг | `~/geoai/services/nginx/nginx.conf` → bind-mount у `/etc/nginx/nginx.conf` (файл, не тека) |
| Сертифікати | volume `geoai_certs` → `/etc/letsencrypt`, оновлює `geoai-certbot-renewer-1` |
| Мережа | `geoai_web`; DNS Docker `resolver 127.0.0.11` на рівні `http {}` |
| Резервні копії | `~/nginx-backups/` на сервері (ніколи не поруч із `nginx.conf`) |
| Тести | [`tests/check_config.sh`](tests/check_config.sh), [`tests/smoke.sh`](tests/smoke.sh) |

### Карта маршрутів (vhost `geohydroai.org`, `listen 443`)

| Шлях | Куди | Репозиторій сервісу |
|---|---|---|
| `/` , `/assets/`, `/index.html` | SPA v1, `/var/www/geoai-front/current` | geoai-frontend (v1) |
| `/app/` | SPA v2, `/var/www/geoai-front/v2/current` | geoai-frontend-v2 |
| `/api/v1/`, `/api/schema`, `/api/docs` | `platform-api:8000` (Django) | geoai-django |
| `/api/v1/internal/` | **403** | — |
| `/api/` (решта) | `api:8000` (FastAPI GeoHydroAI-V2) | GeoHydroAI-V2 |
| `/v1/hydro`, `/llms.txt`, `/llms-full.txt` | `hydro-api:8000` | data_dnipro_h_q (`~/hydro`) |
| `/v1/hydro/admin/` | **403** | — |
| `/tiles/` | `hydro-tiles:3000` (Martin, MVT з PostGIS `geohydro.hydro`) | data_dnipro_h_q |
| `/catalog/` | `catalog-api:8095` (лише GET/HEAD) | geohydroai-knowledge-graph |
| `/tc/` (+ кеш `/tc/singleband/`, `/tc/rgb/`) | `terracotta:5000` | GeoHydroAI-V2 |
| `/dem/`, `/_dash-*` | `dash:8050` | GeoHydroAI-V2 |
| `/reports/` | статика `/var/www/geoai-reports/` | — |
| `/.*` (приховані файли) | **404** | — |

## 2. Правила (для людей і агентів)

1. **Змінювати маршрути — тільки в `services/nginx/nginx.conf` GeoHydroAI-V2.** Не створюйте `nginx-*.conf`,
   `*.snippets.conf` чи інструкції «вставте цей блок» в інших репозиторіях. Сервіс, якому потрібен маршрут,
   описує в своєму README лише *що* йому потрібно (шлях, upstream, ліміти) і посилається сюди.
2. **Upstream до контейнера — завжди динамічний**, інакше після перестворення контейнера (нова IP) буде 502:
   - або `set $x_upstream http://<container>:<port>; proxy_pass $x_upstream;`
   - або `upstream { zone <name> 64k; server <container>:<port> resolve; }`
   Ніколи `server <container>:<port>;` без `resolve`.
3. **`proxy_set_header` успадковується лише якщо в location немає жодного власного.** Якщо задаєте хоч один —
   задайте всі потрібні (`Host`, `X-Forwarded-For`, `X-Forwarded-Proto`, `X-Real-IP`).
4. **Сервер має `proxy_buffering off`.** Location з `proxy_cache` мусить мати `proxy_buffering on`.
5. **Секрети не в конфігу і не в git.** Ключі API додає бекенд (Django-проксі), не nginx і не браузер.
6. **Без попереджень:** `nginx -t` має бути чистим (`tests/check_config.sh` падає на `[warn]`).
7. **Резервні копії — лише в `~/nginx-backups/`**, ім'я `<звідки>__nginx.conf.<мітка>`.

## 3. Як змінити конфіг (порядок обов'язковий)

```bash
# 1. гілка від main, правка лише services/nginx/nginx.conf
git switch -c infra/nginx-<що> origin/main
# 2. перевірка в тимчасовому контейнері (той самий образ, сертифікати, мережа)
services/nginx/tests/check_config.sh
# 3. PR → review → merge у main
```

**⚠️ Автодеплой.** Пуш у `main` запускає `.github/workflows/deploy_aws.yml`, який на сервері робить
`git reset --hard origin/main` у `~/geoai` і `docker compose up -d`. Тобто:
- будь-яка незакомічена правка в `~/geoai` (зокрема `nginx.conf`) **буде стерта**;
- перед злиттям переконайтесь, що `git -C ~/geoai status` чистий, або закомітьте зміни в гілку;
- після деплою, якщо `nginx.conf` змінився: **`docker restart geoai-nginx`** (не `reload`!) і
  `services/nginx/tests/smoke.sh`. `git reset` записує змінений файл як новий inode, а bind-mount файлу тримає
  старий — `nginx -s reload` перечитає **старий** конфіг. compose сам nginx не перезапускає.

Термінова правка прямо на сервері (тільки якщо сайт лежить):
```bash
cp ~/geoai/services/nginx/nginx.conf ~/nginx-backups/geoai_services_nginx__nginx.conf.bak-$(date +%Y%m%d%H%M%S)
# правка на місці (той самий inode — файл змонтовано): редактор або `cat new > nginx.conf`, НЕ mv
docker exec geoai-nginx nginx -t && docker exec geoai-nginx nginx -s reload
services/nginx/tests/smoke.sh
# і того ж дня — коміт цієї правки в main через PR (інакше наступний деплой її зітре)
```

## 4. Перевірка

- `services/nginx/tests/check_config.sh [файл]` — `nginx -t` у тимчасовому контейнері, падає на помилках і попередженнях.
- `services/nginx/tests/smoke.sh [https://geohydroai.org]` — очікувані коди всіх публічних маршрутів
  (401 = маршрут є, потрібен ключ/логін; 204 на тайлі = порожній тайл, не помилка).
- Додаючи маршрут — додайте рядок у `smoke.sh`.
