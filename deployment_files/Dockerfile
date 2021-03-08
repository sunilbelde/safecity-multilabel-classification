FROM python:3.7-slim
RUN python3 --version

RUN pip3 --version


RUN apt-get update && apt-get -y --no-install-recommends install libgomp1

COPY . app
WORKDIR app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt 
EXPOSE 5000
ENTRYPOINT [ "python" ] 
CMD [ "app.py" ] 