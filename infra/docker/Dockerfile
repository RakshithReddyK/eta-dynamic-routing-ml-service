FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY src ./src
COPY data ./data
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "src.eta_routing.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]