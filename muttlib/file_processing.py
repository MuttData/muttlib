from concurrent.futures import Future, ProcessPoolExecutor
import functools
import glob
import inspect
import logging
from multiprocessing import Process, Queue, cpu_count
import os
from pathlib import Path
import time

READY_PREFIX = "ready"
DONE_PREFIX = "done"
FAIL_PREFIX = "fail"

STAGING_PREFIX = "stg"

logger = logging.getLogger(__name__)


class FileNotReady(Exception):
    """Should be raised when filename does not start with READY_PREFIX."""


class RenameByResult:
    """Rename given file based on result to avoid reprocessing.

    Rename input file based on the value passed to set_result at exit.
    """

    def __init__(
        self,
        fn="",
        fn_arg=None,
        ok_unless_exception=True,
        ready_prefix=READY_PREFIX,
        done_prefix=DONE_PREFIX,
        fail_prefix=FAIL_PREFIX,
    ):
        """Setup instance to work as either contextmanager or decorator.

        Either fn or decorate_arg mut be provided

        Args:
            fn (str, Path): Directory from which to read files.
            fn_arg (str): When used as decorator this name of the argument from which the path will
                          be taken.
            ready_prefix (str): Preffix to check if the file is ready.
            done_prefix (str): Preffix to set if call result was successful.
            fail_prefix (str): Preffix to set if call failed.
        """
        if not (fn or fn_arg):
            raise ValueError("Either fn or decorate_arg mut be provided.")
        self.fn = Path(fn)
        self.fn_arg = fn_arg
        self.ok_unless_exception = ok_unless_exception
        self.ready_prefix = ready_prefix
        self.done_prefix = done_prefix
        self.fail_prefix = fail_prefix
        self.result = None
        self._t0 = None

    @property
    def clean_fn(self):
        rv = self.fn.name
        if self.fn.name.startswith(self.ready_prefix):
            rv = rv[len(self.ready_prefix) + 1 :]
        return rv

    def _prefixed_fn(self, prefix):
        return self.fn.parent / f"{prefix}_{self.clean_fn}"

    @property
    def done_fn(self):
        return self._prefixed_fn(self.done_prefix)

    @property
    def fail_fn(self):
        return self._prefixed_fn(self.fail_prefix)

    def __enter__(self):
        """Ensure that the given file is ready."""
        logger.info(f"Processing {self.fn}")
        self._t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Rename input file based on result."""
        if self.result or (self.ok_unless_exception and not exc_val):
            result_fn = self.done_fn
            logger.info(f"DONE ({round(time.time() - self._t0, 3)}s): {self.fn}")
        else:
            result_fn = self.fail_fn
            if exc_val:
                logger.error(f"ERROR: {self.fn}", exc_info=(exc_type, exc_val, exc_tb))
            else:
                logger.warning(f"FAIL: {self.fn}")

        self.fn.rename(result_fn)

    def set_result(self, result):
        """Set attribute to indicate if task was successfull.

        Note that in some cases and empty result (such as [] or {}) can
        be valid but it will be converted into False and the task for
        this file will be considered as failed. In such cases return
        smth like {None: []} that can be easily discarded down the line.
        """
        self.result = bool(result)

    def __call__(self, func):
        sig = inspect.signature(func)
        if self.fn_arg is None:
            raise ValueError(
                "The constructor arg fn_arg must be passed when using as decorator."
            )
        if self.fn_arg not in sig.parameters:
            raise ValueError(
                f"Decorated function doesn't have '{self.fn_arg}' argument."
            )

        @functools.wraps(func)
        def wrapper_decorator(*args, **kwargs):
            self.fn = Path(sig.bind(*args, **kwargs).arguments[self.fn_arg])
            with self as rbr:
                rv = func(*args, **kwargs)
                rbr.set_result(rv)
            return rv

        return wrapper_decorator


class DummyPoolExecutor:
    """Dummy class to emulate multiprocessing."""

    def __init__(self, *args, **kwargs):
        """Keep args and do nothing."""
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        """Start ContextManger."""
        return self

    def __exit__(self, *args):
        """Exit ContextManger."""

    def submit(self, fn, *args, **kwargs):
        """Return dummy future with function return as result."""
        future = Future()
        future.set_result(fn(*args, **kwargs))
        return future


class IsolatedProcessFailedBadly(Exception):
    """Exception raised when the isolated process fails badly."""


def isolate_bad_failures(func, enabled=True):
    """Decorator to isolate code in a separate process.

    This decorator provides a safe way to run code that may fail badly without killing the parent.
    (Fail badly meaning be killed via SIGKILL or suffer segmentation faults).

    If the function runs inside a process based execution pool consider using theads since this
    decorator when (enabled) already solves the GIL issue.

    Args:
        func (function): Function to decorate.
        enabled (bool; True): Enable or disable the isolation.
    """
    if not enabled:
        return func

    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):

        q = Queue()

        def h(q, func, *args, **kwargs):
            try:
                qv = func(*args, **kwargs)
            except Exception as e:  # pylint:disable=W0703
                qv = e
            q.put(qv)

        p = Process(target=h, args=(q, func, *args), kwargs=kwargs)
        p.start()
        p.join()
        if q.empty():
            raise IsolatedProcessFailedBadly(
                "Process ended badly check for kills and segfaults."
            )
        else:
            res = q.get()
            if isinstance(res, Exception):
                raise res
        return res

    return wrapper_decorator
