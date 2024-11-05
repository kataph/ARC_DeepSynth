# def sv(I):
#     x1 = objects(I)(T)(F)(T)
#     x2 = rbind(shoot)(DOWN)
#     x3 = compose(x2)(center)
#     x4 = fork(recolor)(color)(x3)
#     x5 = mapply(x4)(x1)
#     O = paint(I)(x5)
#     return O

# from reformat_ARC_DSL import functions
# from ARC_constants import *
# objects = functions["objects"]; rbind=functions["rbind_2"]; 
# compose=functions["compose"]; fork=functions["fork"]; 
# mapply = functions["mapply_cf"]; paint = functions["paint"]; 
# color=functions["color"]; recolor = functions["recolor"]; 
# center = functions["center"]; shoot = functions["shoot"]
# I = ((1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))
# print(sv(I))

# x1 = objects(I)(T)(F)(T)
# x2 = rbind(shoot)(DOWN)
# x3 = compose(x2)(center)
# x4 = fork(recolor)(color)(x3)
# x5 = mapply(x4)(x1)
# O = paint(I)(x5)

# print("--------------------")
# print(I)
# print(DOWN)
# print(shoot((0,0))(DOWN))
# print(x1)
# print(O)
# print("--------------------")

import errno
import os
import signal
import functools

class TimeoutError(Exception):
    pass

from multiprocessing import Queue, Process

def timeout_windows(seconds, action=None):
    """Calls any function with timeout after 'seconds'.
       If a timeout occurs, 'action' will be returned or called if
       it is a function-like object.
    """
    def handler(queue, func, args, kwargs):
        queue.put(func(*args, **kwargs))
    def decorator(func):
        def wraps(*args, **kwargs):
            q = Queue()
            p = Process(target=handler, args=(q, func, args, kwargs))
            p.start()
            p.join(timeout=seconds)
            if p.is_alive():
                p.terminate()
                p.join()
                print("Termin")
                # if hasattr(action, '__call__'):
                #     return action()
                # else:
                #     return action
            else:
                return q.get()
        return wraps
    return decorator

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator

import time
@timeout_windows(seconds = 3)
def long_function():
    time.sleep(2)

long_function()