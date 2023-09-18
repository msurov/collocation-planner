from multiprocessing import Process, Pool, Queue, Value, current_process
from multiprocessing.connection import wait
from select import select
import numpy as np
from time import sleep, time, process_time


def find_in_parallel(fun : callable, nproc, *args):

    def f(q : Queue):
        np.random.seed(int(time() * 1e+6) & 0xFFFFFFFF)
        res = fun(*args)
        q.put(res)

    q = Queue()
    tasks = [Process(target=f, args=(q,)) for i in range(nproc)]
    for t in tasks: t.start()
    value = None

    while True:
        value = q.get(True)
        if value is not None:
            break
        t = Process(target=f, args=(q,))
        t.start()
        tasks.append(t)
        tasks = filter(lambda t: t.is_alive(), tasks)
        tasks = list(tasks)

    for t in tasks: t.kill()
    return value

