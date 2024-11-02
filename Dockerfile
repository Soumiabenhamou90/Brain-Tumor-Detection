
FROM python:3.10.6
WORKDIR /app
RUN apt-get update
RUN apt-get install \
  'ffmpeg'\
  'libsm6'\
  'libxext6'  -y
COPY . .
RUN pip install -r requirements.txt
RUN pip install .
EXPOSE 8080
CMD uvicorn brainTumorDetection.API.predictTumor:app
