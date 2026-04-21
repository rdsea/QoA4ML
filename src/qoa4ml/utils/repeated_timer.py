import time
from threading import Event, Thread

from qoa4ml.utils.logger import qoa_logger


class RepeatedTimer:
    """Repeat `function` every `interval` seconds.

    Exceptions raised by `function` are logged and swallowed so that a single
    faulty tick does not silently kill the daemon thread; the scheduler
    continues to invoke `function` on every subsequent interval.
    """

    def __init__(self, interval, function, *args, **kwargs):
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.start = time.time()
        self.event = Event()
        self.thread = Thread(target=self._target, daemon=True)
        self.thread.start()

    def _run_once(self) -> None:
        try:
            self.function(*self.args, **self.kwargs)
        except Exception as error:
            error_type = type(error).__name__
            qoa_logger.exception(
                f"RepeatedTimer tick failed ({error_type}); continuing schedule"
            )

    def _target(self):
        self._run_once()
        while not self.event.wait(self._time):
            self._run_once()

    @property
    def _time(self):
        return self.interval - ((time.time() - self.start) % self.interval)

    def stop(self):
        self.event.set()
        self.thread.join()
