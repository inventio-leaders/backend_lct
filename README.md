

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
DATABASE=postgresql+asyncpg://user:password@localhost/dbname
ADMIN_PASSWORD=your_admin_password
SECRET_KEY=your_secret_key
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
