FROM python:3.7

RUN pip install Pillow
RUN pip install tqdm
RUN pip install numpy
RUN pip install tensorflow
RUN pip install matplotlib

RUN pip install jupyterlab

RUN mkdir /app
WORKDIR /app
# ADD . /app


EXPOSE 8888

CMD jupyter lab --ip=0.0.0.0 --port=8888 --allow-root
# CMD bash
# CMD ["python", "/app/main.py"]