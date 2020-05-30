FROM python:3.8
RUN python -m pip install grpcio
RUN python -m pip install grpcio-tools

RUN mkdir /app
ADD . /app
WORKDIR /app
# 
# RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./memegenerator.proto

# CMD python /app/grpc_client.py
CMD ["python", "grpc_client.py"]
