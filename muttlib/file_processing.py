
from concurrent.futures import ProcessPoolExecutor
import glob
import logging
from multiprocessing import cpu_count
import os
import time


READY_PREFIX = 'ready'
DONE_PREFIX = 'done'
FAIL_PREFIX = 'fail'

STAGING_PREFIX = 'stg'


class FileNotReady(Exception):
    """Should be raised when filename does not start with READY_PREFIX."""

    pass


class RenameByResult:
    """Rename given file based on result to avoid reprocessing.

    Rename input file based on the value passed to set_result at exit.
    """

    def __init__(self, fn, ready_prefix=READY_PREFIX, done_prefix=DONE_PREFIX, fail_prefix=FAIL_PREFIX):
        """Setup."""
        self.fn = fn
        self.ready_prefix = ready_prefix
        self.done_prefix = done_prefix
        self.fail_prefix = fail_prefix
        self.result = None
        self.base_fn = os.path.basename(self.fn)

    def __enter__(self):
        """Ensure that the given file is ready."""
        logging.info(f"Processing {self.fn}")
        if not self.base_fn.startswith(self.ready_prefix):
            self.fail_fn = f'{self.fail_prefix}_{self.base_fn}'
            raise FileNotReady()
        else:
            self.clean_fn = self.base_fn[len(self.ready_prefix) + 1:]
        self.done_fn = self.done_prefix + self.base_fn[len(self.ready_prefix):]
        self.fail_fn = self.fail_prefix + self.base_fn[len(self.ready_prefix):]
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Rename input file based on result."""
        if self.result:
            result_fn = self.done_fn
            logging.info(f"DONE ({round(time.time() - self.t0, 3)}s): {self.fn}")
        else:
            result_fn = self.fail_fn
            if exc_val:
                logging.error(f"ERROR: {self.fn}", exc_info=(exc_type, exc_val, exc_tb))
            else:
                logging.warning(f"FAIL: {self.fn}")

        result_fn = os.path.join(os.path.dirname(self.fn), result_fn)
        os.rename(self.fn, result_fn)

    def set_result(self, result):
        """Set attribute to indicate if task was successfull.

        Note that in some cases and empty result (such as [] or {}) can be
        valid but it will be converted into False and the task for this file
        will be considered as failed.
        In such cases return smth like {None: []} that can be easily
        discarded down the line.
        """
        self.result = bool(result)


def get_new_files(in_dir, only_ready=False, ready_prefix=READY_PREFIX):
    """Get files ready for processing."""
    filter_prefix = ready_prefix if only_ready else ''
    fns = glob.glob(os.path.join(in_dir, filter_prefix + '*'))
    return fns


def process_new_files(proc_func, in_dir, **extra_data):
    """Process files that match ready prefix.

    Process files that match ready prefix the given function and rename them
    based on the result.
    """
    for fn in get_new_files(in_dir):
        with RenameByResult(fn) as rbr:
            res = proc_func(fn, rbr, **extra_data)
            rbr.set_result(res)


def worker_wrapper(proc_func, fn, use_rbr=False, **extra_data):
    """Wrap proc_func to add file rename."""
    if use_rbr:
        with RenameByResult(fn) as rbr:
            res = proc_func(fn, rbr, **extra_data)
            rbr.set_result(res)
    else:
        res = proc_func(fn, **extra_data)
    return res


class DummyPoolExecutor:
    """Dummy class to emulate multiprocessing."""

    class DummyFuture:
        """Emulation of concurrent.futures.Future."""

        def __init__(self, result):
            """Set dummy future result with given arg."""
            self.result = result

        def result():
            """Emulate concurrent.futures.Future().result."""
            return self.result

    def __init__(self, *args, **kwargs):
        """Keep args and do nothing."""
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        """Start ContextManger."""
        return self

    def __exit__(self, *args):
        """Exit ContextManger."""
        pass

    def submit(self, fn, *args, **kwargs):
        """Return dummy future with function return as result."""
        return self.DummyFuture(fn(*args, **kwargs))


def process_new_files_parallel(proc_func, in_dir, workers=None, only_ready=False,**extra_data):
    """Process files that match ready prefix in parallel.

    Process files that match ready prefix the given function and rename them
    based on the result.
    By default the number of workers is equal to the number of cores.

    TODO:
    - Add selector for pool type (process or thread). Threads would be the
      preffered option if proc_func frees the GIL either by using Numba or
      Cython.
    """
    if workers is None:
        workers = cpu_count()

    if workers > 0:
        pool = ProcessPoolExecutor(max_workers=workers)
        logging.info(f"Using process {workers} workers for file processing.")
    else:
        pool = DummyPoolExecutor()
        logging.info(f"No for file processing.")

    futures = []
    with pool as executor:
        for fn in get_new_files(in_dir, only_ready):
            futures.append(executor.submit(worker_wrapper, proc_func, fn, **extra_data))

    res = [f.result() for f in futures]

    return res
