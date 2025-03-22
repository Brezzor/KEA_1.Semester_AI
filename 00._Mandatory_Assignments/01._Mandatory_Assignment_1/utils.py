import time

class Timer:
    def __init__(self):
        self.time = None

    def __enter__(self):
        self.time = time.time()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        time_passed = time.time() - self.time
        print(f'** {time_passed:.2f} seconds.')