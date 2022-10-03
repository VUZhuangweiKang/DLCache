import zmq
import time


if __name__ == "__main__":
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5555")
    while True:
        socket.send(b"hello {}".format(time.time()))
        time.sleep(1)