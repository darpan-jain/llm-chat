FROM python:3.10
LABEL authors="darpanjain"

# Copy the current directory contents into the container at /app
COPY .* /app

# Set the working directory to /app
RUN cd /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# Run app.py when the container launches
CMD ["python3", "app/app_chat.py"]
