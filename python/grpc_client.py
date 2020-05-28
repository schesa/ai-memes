from __future__ import print_function
import logging

import grpc

import memegenerator_pb2
import memegenerator_pb2_grpc


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = memegenerator_pb2_grpc.GreeterStub(channel)
        response = stub.GetMemeUrl(
            memegenerator_pb2.MemeRequest(caption='when implementing grpc|is hard'))
    print("Greeter client received url: " + response.url)


if __name__ == '__main__':
    logging.basicConfig()
    run()
