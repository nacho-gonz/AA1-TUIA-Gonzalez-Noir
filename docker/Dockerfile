FROM python:3.10.15-slim

WORKDIR /app

COPY requeriments.txt ./

RUN pip install --no-cache-dir -r requeriments.txt

COPY . .

CMD ["python" , "./inferencia.py"]

