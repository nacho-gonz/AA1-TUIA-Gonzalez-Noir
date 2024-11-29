FROM python:3.10-slim

WORKDIR /app

COPY requeriments.txt ./

RUN pip install --no-cache-dir -r requeriments.txt

COPY . .

VOLUME ["/app/files"]

CMD ["python" , "./inferencia.py"]

