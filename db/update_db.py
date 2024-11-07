# init_db.py
from db.database import engine, Base


def update_database():
    # Создание таблиц
    Base.metadata.create_all(bind=engine)
    print("Таблицы созданы.")


if __name__ == '__main__':
    update_database()
