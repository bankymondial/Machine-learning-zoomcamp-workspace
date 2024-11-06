# 5. Deploying Machine Learning models 

We'll use the same model we trained and evaluated
previously - the churn prediction model. Now we'll
deploy it as a web service.

## 5.1 Intro / Session overview

* What we will cover this week

## 5.2 Saving and loading the model

* Saving the model to pickle
* Loading the model from pickle
* Turning our notebook into a Python script

## 5.3 Web services: introduction to Flask

* Writing a simple ping/pong app
* Querying it with `curl` and browser

## 5.4 Serving the churn model with Flask

* Wrapping the predict script into a Flask app
* Querying it with `requests` 
* Preparing for production: gunicorn
* Running it on Windows with waitress

## 5.5 Python virtual environment: Pipenv

* Dependency and environment management
* Why we need virtual environment
* Installing Pipenv
* Installing libraries with Pipenv
* Running things with Pipenv

## 5.6 Environment management: Docker

* Why we need Docker
* Running a Python image with docker
* Dockerfile
* Building a docker image
* Running a docker image

*docker run -it --rm python:3.8.12-slim
*exit (Ctrl+D)
*docker run -it --rm --entrypoint=bash python:3.8.12-slim
*apt-get update
*apt-get install wget
root@1e47eee68b20:/# mkdir test
root@1e47eee68b20:/# cd test/
root@1e47eee68b20:/test# ls
root@1e47eee68b20:/test# pwd/test
root@1e47eee68b20:/test# pip install pipenv
*docker run -it --rm --entrypoint=bash python:3.8.12-slim
root@b54a08bb7d7e:/# mkdir app
root@b54a08bb7d7e:/# cd app/
root@b54a08bb7d7e:/app# ls
exit

docker build -t zoomcamp-test .
docker run -it --rm --entrypoint=bash zoomcamp-test
ls

include RUN pipenv install --system --deploy into Dockerfile document
exit

docker build -t zoomcamp-test .
docker run -it --rm --entrypoint=bash zoomcamp-test
ls
exit

add COPY ["predict.py", "model_C=1.0.bin", "./"] to Dockerfile document
then, run docker build -t zoomcamp-test .

docker run -it --rm --entrypoint=bash zoomcamp-test
ls

gunicorn -bind=0.0.0.0:9696 predict:app

## 5.7 Deployment to the cloud: AWS Elastic Beanstalk (optional)

* Installing the eb cli
* Running eb locally
* Deploying the model

## 5.8 Summary

* Save models with picke
* Use Flask to turn the model into a web service
* Use a dependency & env manager
* Package it in Docker
* Deploy to the cloud


## 5.9 Explore more

* Flask is not the only framework for creating web services. Try others, e.g. FastAPI
* Experiment with other ways of managing environment, e.g. virtual env, conda, poetry.
* Explore other ways of deploying web services, e.g. GCP, Azure, Heroku, Python Anywhere, etc
