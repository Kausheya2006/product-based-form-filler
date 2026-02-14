# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

RUN pip install --upgrade pip
RUN pip install --no-cache-dir torch --default-timeout=1000

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Set python path so modules can be found
ENV PYTHONPATH=/app

# Default command: Sleep indefinitely to keep container running
# User will 'exec' into this to run the CLI
CMD ["tail", "-f", "/dev/null"]
