FROM python:3.9

WORKDIR /software/

COPY requirements.txt /software/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /software/

EXPOSE 8501

ENV PYTHONUNBUFFERED=1

CMD ["streamlit", "run", "app.py"]