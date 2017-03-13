import zmq
import sys
import threading
import time
from random import randint, random

__author__ = "Felipe Cruz <felipecruz@loogica.net>"
__license__ = "MIT/X11"

def tprint(msg):
    """like print, but won't get newlines confused with multiple threads"""
    sys.stdout.write(msg + '\n')
    sys.stdout.flush()

class ClientTask(threading.Thread):
    """ClientTask"""
    def __init__(self, id):
        self.id = id
        threading.Thread.__init__ (self)

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.DEALER)
        identity = u'client-%d' % self.id
        socket.identity = identity.encode('ascii')
        socket.connect('tcp://localhost:5570')
        
        print('Client %s started' % (identity))
        poll = zmq.Poller()
        poll.register(socket, zmq.POLLIN)
        reqs = 0
        identity = u'worker-%d' % self.id
        worker = identity.encode('ascii')
        while True:
            reqs = reqs + 1
            print('Req #%d sent to %s..' % (reqs, worker))
            reqs_str = 'request #%d' % (reqs)
        
            socket.send_multipart([worker, b'', reqs_str.encode('ascii')])
            time.sleep(5)
            for i in range(2):
                sockets = dict(poll.poll(1000))
                if socket in sockets:
                    rcv_identity,  msg = socket.recv_multipart()
                    tprint('Client %s received: %s - %s' % (socket.identity ,rcv_identity, msg))

        socket.close()
        context.term()

class ServerTask(threading.Thread):
    """ServerTask"""
    def __init__(self):
        threading.Thread.__init__ (self)

    def run(self):
        context = zmq.Context()
        frontend = context.socket(zmq.ROUTER)
        frontend.bind('tcp://*:5570')

        backend = context.socket(zmq.DEALER)
        backend.bind('inproc://backend')

        workers = []
        for i in range(2):
            worker = ServerWorker(context, i)
            worker.start()
            workers.append(worker)

        zmq.proxy(frontend, backend)

        frontend.close()
        backend.close()
        context.term()

class ServerWorker(threading.Thread):
    """ServerWorker"""
    def __init__(self, context, idx):
        threading.Thread.__init__ (self)
        self.context = context
        self.idx = idx

    def run(self):
        worker = self.context.socket(zmq.DEALER)
        identity = u'worker-%d' % self.idx
        worker.identity = identity.encode('ascii')
        worker.connect('inproc://backend')
        tprint('Worker started')
        aa = u'client-0'
        while True:
            ident, empty, msg, x1 = worker.recv_multipart()
            tprint('Worker [%d] received [%s, %s, %s, %s]' % (self.idx, ident, empty, msg, x1 ))
            replies = 1#randint(0,4)
            msg += "  Reply from worker [%d]" % (self.idx)
            for i in range(replies):
                time.sleep(1. / (randint(1,10)))
                worker.send_multipart([aa.encode('ascii'), b'', msg.encode('ascii')])

        worker.close()

def main():
    """main function"""
    server = ServerTask()
    server.start()
    for i in range(2):
        client = ClientTask(i)
        client.start()

    server.join()

if __name__ == "__main__":
    main()
