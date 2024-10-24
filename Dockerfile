# the official python slim image
FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0
    # we should install system depedencies for OpenCV, because these are libraries that are not written 
    # in Python cuz openCV relies on certain system libraries to handle graphical operations or other lower-level
    # tasks. These libraries need to be installed at the OS level
    
# set the working directory in the container
WORKDIR /app

EXPOSE 8000 8001

# copy the requirements txt 
# we install the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the entire project into the container
COPY . .

# run the FastAPI app using Uvicorn server
CMD ["uvicorn", "app.main:myapp", "--host", "0.0.0.0", "--port", "8000"]
