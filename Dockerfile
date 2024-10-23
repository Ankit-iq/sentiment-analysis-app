FROM python:3.12-slim

WORKDIR /app

# Install necessary libraries
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader punkt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "Sentiment_Analyzer_app.py"]
