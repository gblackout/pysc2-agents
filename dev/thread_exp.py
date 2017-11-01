"""
some experiments with python threading lib
"""
import numpy as np
from multiprocessing import Process, Pipe, Array
import sys
import time


def fff():
    sys.stdout.flush()


def work(*args, **kwargs):
    a = Agent(*args, **kwargs)
    a.run()


class Agent():

    def __init__(self, agent_id, buf, buf_shape, c):
        self.agent_id = agent_id
        self.buf = np.reshape(np.frombuffer(buf, dtype=np.float32), buf_shape)
        self.bias = 0
        self.c = c

    def run(self):
        while True:
            self.c.recv()

            self.buf[self.agent_id, :, :, :, :] = self.agent_id+self.bias
            self.bias += 1

            self.c.send(True)


def mm():
    num_agent = 2
    batch_size = 5
    num_stack = 4
    h, w, c = 160, 210, 3
    buf_shape = (num_agent * batch_size, num_stack, h, w, c)

    shared_buf = Array('f', num_agent * batch_size * num_stack * h * w * c, lock=False) # not process-safe anymore
    buf = np.reshape(np.frombuffer(shared_buf, dtype=np.float32), buf_shape)

    p_ls, c_ls = [], []
    for p_ind in xrange(num_agent):

        c1, c2 = Pipe()
        c_ls.append(c1)

        p = Process(target=work, args=(p_ind, shared_buf, buf_shape, c2))

        p.daemon = True
        p.start()
        p_ls.append(p)

    st = time.time()
    for _ in xrange(1000):
        for b in xrange(batch_size):

            for c in c_ls:
                c.send(True)

            _ = [c.recv() for c in c_ls]

    for p in p_ls:
        p.terminate()

    print time.time() - st







     







# def main():
#
#     num_agent = 6
#     batch_size = 5
#     num_stack = 4
#     h, w, c = 160, 210, 3
#
#     buf = np.zeros((num_agent*batch_size, num_stack, h, w, c))
#
#     worker_threads = []
#     queue_ls = []
#     for t_ind in xrange(num_agent):
#         q = Queue.Queue()
#         t = Agent(t_ind, buf, q)
#
#         worker_threads.append(t)
#         queue_ls.append(q)
#
#         t.start()
#
#     import time
#     st = time.time()
#
#     for b in xrange(batch_size):
#
#         for t_ind in xrange(num_agent):
#             offset = t_ind * batch_size + b
#             queue_ls[t_ind].put(offset)
#
#         for q in queue_ls:
#             q.join()
#
#     print time.time() - st

    # print buf[:4]
    # print '\n', '-'*50, '\n'
    # print buf[4:]

if __name__ == '__main__':
    # main()
    mm()