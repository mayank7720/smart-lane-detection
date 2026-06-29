# Use a lightweight official Python image
FROM python:3.11-slim

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy dependency definition
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY . .

# Expose the application port
EXPOSE 5000

# Set environment variables
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Install waitress for production WSGI server
RUN pip install --no-cache-dir waitress

# Run the app using waitress
CMD ["python", "-c", "from waitress import serve; from run_app import app; serve(app, host='0.0.0.0', port=5000, threads=4)"]
