# ==============================
# 🐳 Smart Health Risk Predictor Dockerfile
# ==============================

# 1️⃣ Base image - use lightweight official Python
FROM python:3.11-slim

# 2️⃣ Set working directory
WORKDIR /app

# 3️⃣ Copy all project files
COPY . /app

# 4️⃣ Upgrade pip and install dependencies
RUN pip install --upgrade pip

# If you have a requirements.txt file, use this:
RUN pip install -r requirements.txt

# 5️⃣ Expose Streamlit default port
EXPOSE 8501

# 6️⃣ Set environment variables (disable Streamlit telemetry)
ENV STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_PORT=8501 \
    PYTHONUNBUFFERED=1

# 7️⃣ Run Streamlit app
CMD ["streamlit", "run", "streamlit_app.py"]
