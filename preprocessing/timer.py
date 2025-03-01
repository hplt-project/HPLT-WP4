import time


class Timer:
    def __init__(self, max_time):
        self.max_time = max_time
        self.start_time = time.time()

    def time_remaining(self):
        return self.max_time - (time.time() - self.start_time)

    def has_time_remaining(self):
        return self.time_remaining() > 0
