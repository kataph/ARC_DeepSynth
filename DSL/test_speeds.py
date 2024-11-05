# test speeds
from functools import wraps
import time
import os
from typing import Any


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper
def timeitlogs(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds\n')
        with open("DSL\\performances.txt", "a") as fo:
            fo.write(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds\n')
        return result
    return timeit_wrapper

@timeit
def loop_with_ors_ands(n_iter: int) -> None:
    for i in range(n_iter):
        b = False
        b = b or (i//2 == 0 and i//3 == 0 and i//4 == 0 and i//5 == 0)
        b = b or (i//3 == 0 and i//3 == 0 and i//4 == 0 and i//5 == 0)
        b = b or (i//4 == 0 and i//3 == 0 and i//4 == 0 and i//5 == 0)
        b = b or (i//5 == 0 and i//3 == 0 and i//4 == 0 and i//5 == 0)
        b = b or (i//6 == 0 and i//3 == 0 and i//4 == 0 and i//5 == 0)
@timeit
def loop_with_ifs_ands(n_iter: int) -> None:
    for i in range(n_iter):
        b = False
        if (i//2 == 0 and i//3 == 0 and i//4 == 0 and i//5 == 0):
            b = True
            continue
        if (i//3 == 0 and i//3 == 0 and i//4 == 0 and i//5 == 0):
            b = True
            continue
        if (i//4 == 0 and i//3 == 0 and i//4 == 0 and i//5 == 0):
            b = True
            continue
        if (i//5 == 0 and i//3 == 0 and i//4 == 0 and i//5 == 0):
            b = True
            continue
        if (i//6 == 0 and i//3 == 0 and i//4 == 0 and i//5 == 0):
            b = True
@timeit
def loop_with_elifs_ands(n_iter: int) -> None:
    for i in range(n_iter):
        b = False
        if (i//2 == 0 and i//3 == 0 and i//4 == 0 and i//5 == 0):
            b = True
            continue
        elif (i//3 == 0 and i//3 == 0 and i//4 == 0 and i//5 == 0):
            b = True
            continue
        elif (i//4 == 0 and i//3 == 0 and i//4 == 0 and i//5 == 0):
            b = True
            continue
        elif (i//5 == 0 and i//3 == 0 and i//4 == 0 and i//5 == 0):
            b = True
            continue
        elif (i//6 == 0 and i//3 == 0 and i//4 == 0 and i//5 == 0):
            b = True

@timeit
def write_big(line: Any = "This is the text I want to copy 100's of time", 
              chunk_size: int = 1, n_lines: int = 10_000_000, flush: bool = False):
    line = str(line)
    with open( "DSL\\largefile.txt","wt" ) as output_file:
        for n in range(n_lines//chunk_size):
            chunk = line * chunk_size
            output_file.write(chunk)
            if flush:
                output_file.flush() # no effect...
                os.fsync(output_file.fileno()) #questo > triplica il rempo di esecuzion

from time import perf_counter, sleep
if __name__ == "__main__":
    pass
    # write_big(chunk_size=1, n_lines=10_000_000) # 20s O(n)
    # write_big(chunk_size=10, n_lines=10_000_000) # 7s O(n)
    # write_big(chunk_size=100, n_lines=10_000_000) # 7s O(n)

    # write_big(chunk_size=1000, n_lines=10_000_000, flush=True) # 5s O(n)
    # write_big(chunk_size=10000, n_lines=10_000_000) # 5s O(n)
    # write_big(chunk_size=100000, n_lines=10_000_000) # 8s O(n)
    # write_big(chunk_size=1_000_000, n_lines=10_000_000) # 5s O(n)
    # write_big(chunk_size=10_000_000, n_lines=10_000_000) # 10s O(n)
    # write_big(chunk_size=10, n_lines=10_000_000) # 9s O(n)
    # write_big(n_lines=10)
    # write_big(n_lines=100)
    # write_big(n_lines=1000)
    # write_big(n_lines=10000)
    # write_big(n_lines=100000)
    # write_big(n_lines=1_000_000)
    # write_big(n_lines=10_000_000) # 26 s O(n)
    # loop_with_ors_ands(n_iter=10)
    # loop_with_ors_ands(n_iter=100)
    # loop_with_ors_ands(n_iter=1000)
    # loop_with_ors_ands(n_iter=10000)
    # loop_with_ors_ands(n_iter=100000)
    # loop_with_ors_ands(n_iter=1000000)
    # loop_with_ors_ands(n_iter=10000000) # c.a. 7s; O(n_iter)
    # loop_with_ifs_ands(n_iter=10)
    # loop_with_ifs_ands(n_iter=100)
    # loop_with_ifs_ands(n_iter=1000)
    # loop_with_ifs_ands(n_iter=10000)
    # loop_with_ifs_ands(n_iter=100000)
    # loop_with_ifs_ands(n_iter=1000000)
    # loop_with_ifs_ands(n_iter=10000000) # c.a. 5s; O(n_iter)
    # loop_with_elifs_ands(n_iter=10)
    # loop_with_elifs_ands(n_iter=100)
    # loop_with_elifs_ands(n_iter=1000)
    # loop_with_elifs_ands(n_iter=10000)
    # loop_with_elifs_ands(n_iter=100000)
    # loop_with_elifs_ands(n_iter=1000000)
    # loop_with_elifs_ands(n_iter=10000000) # c.a. 5s; O(n_iter)