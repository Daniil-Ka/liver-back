# init_db.py
from database import engine, Base

# Создание таблиц
Base.metadata.create_all(bind=engine)
print("Таблицы созданы.")
