FROM python:3.9.6

WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

EXPOSE 80 6006 8501

# Command to run the application
CMD ["streamlit", "run", "main.py"]