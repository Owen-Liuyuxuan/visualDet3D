from time import time
from functools import wraps
import torch
def profile(name, profile_start=0, profile_end=1):
    def profile_wrapper(func):
        @wraps(func)
        def profiled_func(*args, **kwargs):
            # determine if started           
            if profiled_func.call_time < profile_end:
                torch.cuda.synchronize()
                start_time = time()
                result = func(*args, **kwargs)
                torch.cuda.synchronize()
                dt = time() - start_time
                if profiled_func.call_time >= profile_start:
                    print("{} takes {} seconds at call time {}".format(name, dt, profiled_func.call_time))
                profiled_func.call_time += 1
            else:
                result = func(*args, **kwargs)
            return result

        # initilize call time to be zero
        profiled_func.call_time = 0
        return profiled_func
    return profile_wrapper


class Timer:
    """
        Timer class is a helper class useful for logging time consumption.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.init_time = time()
        self.time_now = self.init_time

    def time_diff_per_n_loops(self):
        time_diff = time() - self.time_now
        self.time_now = time()
        return time_diff

    def total_time(self):
        """
            compute average time over iterations, return in float
        """
        return time() - self.init_time

    def _compute_avg_time(self, iteration):
        return self.total_time() / float(iteration)
    
    def compute_avg_time(self, iteration):
        """
            compute average time over iterations, return in string
        """
        return formatting_time(  self._compute_avg_time(iteration) )

    def _compute_eta(self, current_iter, total_iter):
        """
            compute estimated time to last, return in float
        """
        return self._compute_avg_time(current_iter) * (total_iter - current_iter)

    def compute_eta(self, current_iter, total_iter):
        """
            compute estimated time to last, return in string
        """
        return formatting_time(self._compute_eta(current_iter, total_iter))

def formatting_time(float_time):
    """
    Computes the estimated time as a formatted string as well
    """

    if float_time > 3600: time_str = '{:.2f}h'.format(float_time / 3600);
    elif float_time > 60: time_str = '{:.2f}m'.format(float_time / 60);
    else: time_str = '{:.2f}s'.format(float_time);

    return time_str

if __name__ == '__main__':
    from time import sleep

    timeit = Timer()
    for i in range(10):
        sleep(1)
        print('It takes {} secs per loop.'.format(timeit.time_diff_per_n_loops()))

    print('It takes {} secs per whole script.'.format(timeit.total_time()))

