FROM python:3.7

MAINTAINER gurkansahin08@gmail.com

RUN apt update
RUN apt-get install -y python3-pip python-dev build-essential
RUN apt-get install -y git

COPY . /
WORKDIR /
RUN pip3 install -r requirements.txt

CMD ["python", "text_classification_service/manage.py", "runserver", "0.0.0.0:8000"]

