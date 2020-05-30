FROM python:3.7

RUN pip install django
RUN pip install Pillow
RUN pip install graphene_django
RUN python -m pip install grpcio
RUN python -m pip install grpcio-tools

RUN mkdir /app
ADD . /app
WORKDIR /app

# RUN python manage.py migrate

CMD ["python", "manage.py", "runserver", "0:8080"]