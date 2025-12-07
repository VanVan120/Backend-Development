# Use the official Python 3.10 image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for OpenCV (cv2)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first (for better caching)
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir keeps the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create the temp_uploads directory (since it's ignored in git)
RUN mkdir -p temp_uploads

# Expose port 7860 (Standard for Hugging Face Spaces)
EXPOSE 7860

# Command to run the application
# We use the uvicorn command directly to override the local settings in main.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
