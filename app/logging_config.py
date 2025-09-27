import logging
import os
from logging.handlers import RotatingFileHandler

os.makedirs("logs", exist_ok=True)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

access_logger = logging.getLogger("uvicorn.access")
access_logger.setLevel(logging.INFO)
access_handler = RotatingFileHandler("logs/access.log", maxBytes=10*1024*1024, backupCount=5)
access_handler.setFormatter(formatter)
access_logger.addHandler(access_handler)
access_logger.propagate = False

error_logger = logging.getLogger("uvicorn.error")
error_logger.setLevel(logging.ERROR)
error_handler = RotatingFileHandler("logs/error.log", maxBytes=10*1024*1024, backupCount=5)
error_handler.setFormatter(formatter)
error_logger.addHandler(error_handler)
error_logger.propagate = False

db_logger = logging.getLogger("sqlalchemy.engine")
db_logger.setLevel(logging.INFO)
db_handler = RotatingFileHandler("logs/db.log", maxBytes=10*1024*1024, backupCount=5)
db_handler.setFormatter(formatter)
db_logger.addHandler(db_handler)
db_logger.propagate = False

app_logger = logging.getLogger("app")
app_logger.setLevel(logging.INFO)
app_handler = RotatingFileHandler("logs/app.log", maxBytes=10*1024*1024, backupCount=5)
app_handler.setFormatter(formatter)
app_logger.addHandler(app_handler)
app_logger.propagate = False

print("✅ Логирование настроено!")
