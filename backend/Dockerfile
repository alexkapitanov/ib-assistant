FROM python:3.11-slim
WORKDIR /app

# устанавливаем зависимости
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# копируем исходники и запускаем
COPY backend ./backend
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
