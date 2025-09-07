# Set Python version for the base image.
FROM python:3.12-slim

# Set the working directory
WORKDIR /code

# Copy the requirements file into the working directory
COPY ./requirements.txt code/requirements.txt

# Install the dependencies
RUN pip install -r code/requirements.txt

# Copy the entire project into the working directory
COPY ./app /code/app

# Create logs directory
RUN mkdir -p /code/logs

# ENV WANDB_API_KEY=""
# ENV API_KEYS=""
# ENV WANDB_ORG=""
# ENV WANDB_PROJECT=""
# ENV WANDB_MODEL_NAME=""
# ENV WANDB_MODEL_VERSION=""

# Expose port 8080
EXPOSE 8080

CMD ["fastapi", "run", "app/main.py", "--port", "8080", "--reload"]
