

# Документация по запуску Backend

## Требования

* Python 3.10+
* Docker и Docker Compose
* Git

---

## Клонирование репозитория

```bash
git clone  git@github.com:inventi0/backend_QR.git
cd backend_QR
```


3. Создай файл `.env` в корне проекта и добавь переменные окружения, (у Славы, либо у меня):
   ниже тупо пример
   
```
DATABASE = "postgresql+asyncpg://<username>:<password>@<host>:<port>/<database_name>"
LOGIN_DB = "<db_login>"
PASSWORD_DB = "<db_password>"

ADMIN_PASSWORD = "<admin_password>"

PRIVATE_KEY = "<your_private_key>"

SMTP_HOST = "<smtp_host>"
SMTP_PORT = <smtp_port>

SMTP_USER = "<smtp_user>"
SMTP_PASS = "<smtp_password>"

SMTP_FROM = "<from_email>"
SMTP_TO = "<to_email_1>,<to_email_2>"

SMTP_USE_TLS = <true_or_false>

SMTP_SUBJECT_PREFIX = "<subject_prefix>"
```

## Запуск с Docker

1. Убедись, что Docker и Docker Compose установлены и запущены.

2. Запусти контейнер:

```bash
docker compose up --build
```

3. Backend будет доступен по адресу:

```
http://localhost:8080/docs
```
4. Длч того чтобы выключить:
```bash
docker compose down
```
---

* убедись, что порты не заняты.
