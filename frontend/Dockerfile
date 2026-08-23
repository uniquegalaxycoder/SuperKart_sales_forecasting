FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8501

# BACKEND_URL should be set at `docker run` time to point at the backend's
# forwarded URL, e.g.:
#   docker run -e BACKEND_URL=http://backend-host:8000 -p 8501:8501 superkart-frontend
ENV BACKEND_URL=http://localhost:8000

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
