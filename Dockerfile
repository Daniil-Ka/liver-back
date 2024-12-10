# Базовый образ
FROM python:3.11

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем зависимости для OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Копируем зависимости и устанавливаем их
COPY requirements.txt .
RUN pip install -r requirements.txt

# Копируем всё приложение
COPY . .

# Запуск приложения
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
