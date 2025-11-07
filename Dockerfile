FROM python:3.12-slim

#set working directory
WORKDIR /colonoscopy-triage-azure

#copy requirements file, install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#copy code
COPY app ./app 
COPY client_scripts ./client_scripts
COPY data ./data

CMD ["uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
