import time

class Timer:
    def __init__(self):
        self._startTime = None
        self._totalElapsed = 0.0

    def start(self):
        if self._startTime is not None:
            raise Exception(f'Already started.')
        self._startTime = time.perf_counter()

    def stop(self):
        if self._startTime is None:
            raise Exception(f'Timer is not running.')
        elapsed_time = time.perf_counter() - self._startTime
        self._totalElapsed += elapsed_time
        self._startTime = None
        return elapsed_time

    def reset(self):
        totalTime = self._totalElapsed
        self._totalElapsed = 0.0
        self._startTime = None
        return totalTime
