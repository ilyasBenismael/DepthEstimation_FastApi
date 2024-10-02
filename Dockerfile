# Use the official Python slim image
FROM python:3.9-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0
    # System Dependencies: These are libraries that are not written in Python but are required by Python packages.
    # For example, OpenCV relies on certain system libraries to handle graphical operations or other lower-level
    # tasks. These libraries need to be installed at the OS level, and therefore are specified in the 
    # Dockerfile using apt-get. They are not Python packages and thus cannot be listed in requirements.txt.

    
# Set the working directory in the container
WORKDIR /app

EXPOSE 8000 8001

# Copy requirements.txt into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Run the FastAPI app using Uvicorn server
CMD ["uvicorn", "app.main:myapp", "--host", "0.0.0.0", "--port", "8000"]
