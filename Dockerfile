# Use the official Python base image
#FROM python:3.10.6-buster
FROM tensorflow/tensorflow:2.10.0

#WORKDIR /app
WORKDIR /prod

# First, pip install dependencies
#COPY requirements.txt requirements.txt
#RUN pip install -r requirements.txt
COPY requirements_prod.txt requirements_prod.txt
RUN pip install --no-cache-dir -r requirements_prod.txt

# Then only, install taxifare!
COPY taxifare taxifare
COPY setup.py setup.py
RUN pip install .

COPY Makefile Makefile
RUN make reset_local_files

# Command to run the API when the container starts
#CMD uvicorn taxifare.api.fast:app --host 0.0.0.0


# Use CMD to specify default arguments which can be overridden by Google Cloud Run
CMD uvicorn taxifare.api.fast:app --host 0.0.0.0 --port $PORT
