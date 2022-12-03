import concurrent.futures
import os
import multiprocessing as mp


from functools import wraps
from joblib import Parallel, delayed
import time

def make_parallel(func):
    """
        Decorator used to decorate any function which needs to be parallized with concurrent.futures
        (see https://docs.python.org/3/library/concurrent.futures.html).
        When decorated, the input of the function should be a list in which each element is a instance of input for the normal function.
        You can also pass in keyword arguments seperatley.
        :param func: function
            The instance of the function that needs to be parallelized.
        :return: function
    """

    @wraps(func)
    def wrapper(lst):
        """
        :param lst:
            The inputs of the function in a list.
        :return:
        """
        # the number of threads that can be max-spawned.
        # If the number of threads are too high, then the overhead of creating the threads will be significant.
        # Here we are choosing the number of CPUs available in the system and then multiplying it with a constant.
        # In my system, i have a total of 8 CPUs so i will be generating a maximum of 16 threads in my system.
        number_of_threads_multiple = 1 # You can change this multiple according to you requirement
        number_of_workers = int(os.cpu_count() * number_of_threads_multiple)
        if len(lst) < number_of_workers:
            # If the length of the list is low, we would only require those many number of threads.
            # Here we are avoiding creating unnecessary threads
            number_of_workers = len(lst)

        if number_of_workers:
            if number_of_workers == 1:
                # If the length of the list that needs to be parallelized is 1, there is no point in
                # parallelizing the function.
                # So we run it serially.
                result = [func(lst[0])]
            else:
                # Core Code, where we are creating max number of threads and running the decorated function in parallel.
                result = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=number_of_workers) as executer:
                    bag = {executer.submit(func, i): i for i in lst}
                    for future in concurrent.futures.as_completed(bag):
                        result.append(future.result())
        else:
            result = []
        return result
    return wrapper

# TIMER
def timer(func):
    """Print the runtime of the decorated function"""
    @wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"\n@TIMER {func.__name__!r}: {run_time:.4f} s")
        return value
    return wrapper_timer

# PARALLELIZE
# def parallel(_func=None, *, n_jobs=mp.cpu_count()):
#     def decorator_parallel(func):

#         @wraps(func)
#         def wrapper(arg):
#             print(arg)
#             pool = mp.Pool(n_jobs)
#             out = pool.map(func, arg)
#             pool.close()
#             # NB if I use map arg must be an iterable
#             return out
#         return wrapper

#     if _func is None: return decorator_parallel
#     else: return decorator_parallel(_func)

