FROM continuumio/anaconda3:4.4.0
COPY . /Users/praveen/PycharmProjects/Docker
EXPOSE 5000
WORKDIR /Users/praveen/PycharmProjects/Docker
RUN pip install -r requirements.txt
CMD python Flasgger.py
