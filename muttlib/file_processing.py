"""Utilities to facilitate file processing.

This module provides convenient functions and classes for renaming files after
successful operations and paralellized processing of new files.
"""
from concurrent.futures import Future, ProcessPoolExecutor, wait as wait_futures
import functools
import glob
import inspect
import logging
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
    """Renames given file based on result to avoid reprocessing.

    This class can be used as instance manager or decorator
    input file based on the value passed to set_result at exit.
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
        """Initialize instance.

        Either fn or fn_arg must be provided.

        Parameters
        ----------
        fn : str or Path
            Path to the file to processes.
        fn_arg : str
            When used as decorator this is the name of the argument with the path
            to be renamed.
        ok_unless_exception : bool
            Considers any result successful unless an exception happens.
        ready_prefix : str
            Preffix to check if the file is ready.
        done_prefix : str
            Preffix to set if call result was successful.
        fail_prefix : str
            Preffix to set if call failed.

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
        """Return the name of the file without prefix

        Returns
        -------
        str
            Resulting file name.
        """
        rv = self.fn.name
        if self.fn.name.startswith(self.ready_prefix):
            rv = rv[len(self.ready_prefix) + 1 :]
        return rv

    def _prefixed_fn(self, prefix):
        return self.fn.parent / f"{prefix}_{self.clean_fn}"

    @property
    def done_fn(self):
        """Return the name of the file if the result was successful.

        Returns
        -------
        str
            Resulting file name.
        """
        return self._prefixed_fn(self.done_prefix)

    @property
    def fail_fn(self):
        """Return the name of the file if the result was failure.

        Returns
        -------
        str
            Resulting file name.
        """
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
        """Set attribute to indicate if task was successful.

        Note that in some cases and empty result (such as [] or {}) can
        be valid but it will be converted into False and the task for
        this file will be considered as failed. In such cases return
        smth like {None: []} that can be easily discarded down the line.

        Parameters
        ----------
        result : Any
            Value to be evaluated as `bool` to consider the task successful.
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
    """Dummy class to emulate a Pool Executor."""

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


def get_new_files(in_dir, only_ready=False, ready_prefix=READY_PREFIX):
    """Get files in a directory with a given prefix ready to be for processing.

    Parameters
    ----------
    in_dir : Path or str
        Input directory.
    only_ready : bool
        The second parameter.
    ready_prefix : str
        Prefix to consider the file ready.

    Returns
    -------
    list of Path
        Paths matching the ready prefix.
    """

    filter_prefix = ready_prefix if only_ready else ''
    fns = glob.glob(os.path.join(in_dir, filter_prefix + '*'))
    return fns


def process_new_files(proc_func, in_dir, **extra_data):
    """Process files that match ready prefix.

    Process files that match ready prefix the given function and rename them
    based on the result.

    Parameters
    ----------
    proc_func : callable
        Function to be called for each ready file in in_dir.
        The filename and a RenameByResult instance will be passed as arguments.
    in_dir : str or Path
        Input directory.
    extra_data : dict
        Extra kwargs to pass to the fucntion.
    """
    for fn in get_new_files(in_dir):
        with RenameByResult(fn) as rbr:
            res = proc_func(fn, rbr, **extra_data)
            rbr.set_result(res)


def process_new_files_parallel(func, paths, workers=None, args=None, kwargs=None):
    """Process files that match ready prefix in parallel.

    Process files that match ready prefix the given function and rename them
    based on the result.
    By default the number of workers is equal to the number of cores.

    Parameters
    ----------
    func : callable
        Function to apply.
    paths : list of str
        Paths of files that would be processed by func.
    workers : int
        Max number of workers to spawn. None default to cpu count.
        Passing -1 forces single thread execution (good for debugging).
    args : list or None
        Extra args to length argument list.
    kwargs : dict or None
        Arbitrary kwargs to be passed to the function.

    Returns
    -------
    list of Any
        Returns with the results of applying the function to each file.

    TODO:
    - Add selector for pool type (process or thread). Threads would be the
      preffered option if `fn` frees the GIL either by using Numba or
      Cython.
    """

    if args is None:
        args = []

    if kwargs is None:
        kwargs = {}

    if workers is None:
        workers = os.cpu_count()

    if workers > 0:
        pool = ProcessPoolExecutor(max_workers=workers)
        logger.info(f"Using process {workers} workers for file processing.")
    else:
        pool = DummyPoolExecutor()
        logger.info("Using a single worker.")

    futures = []
    with pool as executor:
        for p in paths:
            futures.append(executor.submit(func, p, *args, **kwargs))

    wait_futures(futures)
    res = [f.result() for f in futures]
    return res
