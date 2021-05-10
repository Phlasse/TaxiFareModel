FROM python:3.8.6-buster

COPY api /api
COPY TaxiFareModel /TaxiFareModel
COPY model.joblib /model.joblib
COPY requirements.txt /requirements.txt
COPY /Users/zastrow/Documents/gcp_keys/phlasselw.json /credentials.json

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT

#gcloud run deploy \
#    --image eu.gcr.io/$PROJECT_ID/$DOCKER_IMAGE_NAME \
#    --platform managed \
#    --region europe-west1 \
#    --set-env-vars "GOOGLE_APPLICATION_CREDENTIALS=/credentials.json"