# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependency lock file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# We use --no-cache-dir to keep the image size small
RUN pip install --no-cache-dir -r requirements.txt

# Install Gunicorn for a production-ready WSGI server
RUN pip install --no-cache-dir gunicorn

# Copy the rest of the application code into the container
COPY . .

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define the command to run the application using Gunicorn
# This will run 4 worker processes to handle incoming requests.
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:5000", "api:app"]