import grpc

import memegenerator_pb2
import memegenerator_pb2_grpc


def get_url(caption, id):
    print(f'request {id} {caption}')
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = memegenerator_pb2_grpc.MemerStub(channel)
        response = stub.GetMemeUrl(
            memegenerator_pb2.MemeRequest(
                caption=caption, memeid=id))
    print("Memer client received url: " + response.url)


if __name__ == '__main__':
    get_url('when implementing grpc|is hard', "102156234")
