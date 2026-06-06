FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# إعداد مستخدم خاص بـ Hugging Face (مهم جداً عشان صلاحيات الكتابة في الداتا بيز)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install Python dependencies (مع نقل الملكية للمستخدم الجديد)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=user . .

# Create data and database directories
RUN mkdir -p data database

# استخدام البورت الخاص بـ Hugging Face
EXPOSE 7860

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]