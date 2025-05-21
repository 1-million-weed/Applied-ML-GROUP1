FROM python:3.9.6

WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# RUN apt-get install ffmpeg

# Copy the rest of the application code into the container
COPY . .

# Command to run the application
CMD ["python3", "-u", "main.py"]