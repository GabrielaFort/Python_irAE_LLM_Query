FROM python:3.11-slim

WORKDIR /docker

# Copy all app code
COPY src ./src
COPY data/irae_data_cleaned.csv ./data/irae_data_cleaned.csv
COPY app.py .

# Copy requirements
COPY docker/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8501

# Run streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
