import zmq
import time


if __name__ == "__main__":
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b'hello')
    socket.connect("tcp://localhost:5555")
    while True:
        topic, data = socket.recv()
        print(data)