node .\node\dynamic_codegen\greeter_client.js
python -m grpc_tools.protoc -I../../ --python_out=. --grpc_python_out=. ../../helloworld.proto
py .\python\helloworld\greeter_server.py
