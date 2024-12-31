# Use Python 3.8 as the base image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the Flask app into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the app's port
EXPOSE 5000

# Start the app
CMD ["python", "app.py"]
