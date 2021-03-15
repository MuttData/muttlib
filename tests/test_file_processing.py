from concurrent.futures import ThreadPoolExecutor, wait
import contextlib
import os
from pathlib import Path
import signal
from tempfile import TemporaryDirectory

import pytest

from muttlib.file_processing import (  # merge_result_dfs,; split_res,
    DONE_PREFIX,
    FAIL_PREFIX,
    READY_PREFIX,
    DummyPoolExecutor,
    RenameByResult,
    process_new_files,
    process_new_files_parallel,
)


def get_ctxmgr_exception_tester(exc=None):
    """Returns context manager to either check for the given exception or do nothing."""
    if exc:
        ctxmgr = pytest.raises(exc)
    else:
        ctxmgr = contextlib.suppress()
    return ctxmgr


@contextlib.contextmanager
def run_rbr_helper_contextmanager(base_fn, expected_fn, expected_exc=None):
    """Implements context manager to perform common checks and cleanup."""
    with TemporaryDirectory() as d:
        d = Path(d)
        fn = d / base_fn
        with open(fn, "wb"):
            pass

        with get_ctxmgr_exception_tester(expected_exc):
            yield fn

        assert (d / expected_fn).exists()


def f_ok():
    return True


def f_fail():
    return False


def f_exception():
    raise Exception()


def f_valueerror():
    raise ValueError()


rbr_fixture = [
    (f_ok, "test", f"{DONE_PREFIX}_test", None, False),
    (f_fail, "test", f"{FAIL_PREFIX}_test", None, False),
    (f_exception, "test", f"{FAIL_PREFIX}_test", Exception, False),
    (f_ok, f"{READY_PREFIX}_test", f"{DONE_PREFIX}_test", None, False),
    (f_fail, f"{READY_PREFIX}_test", f"{FAIL_PREFIX}_test", None, False),
    (f_exception, f"{READY_PREFIX}_test", f"{FAIL_PREFIX}_test", Exception, False),
    (f_ok, "test", f"{DONE_PREFIX}_test", None, True),
    (f_fail, "test", f"{DONE_PREFIX}_test", None, True),
    (f_exception, "test", f"{FAIL_PREFIX}_test", Exception, True),
    (f_ok, f"{READY_PREFIX}_test", f"{DONE_PREFIX}_test", None, True),
    (f_fail, f"{READY_PREFIX}_test", f"{DONE_PREFIX}_test", None, True),
    (f_exception, f"{READY_PREFIX}_test", f"{FAIL_PREFIX}_test", Exception, True),
]


@pytest.mark.parametrize(
    "func, base_fn, expected_fn, expected_exc, ok_unless_exception", rbr_fixture
)
def test_rbr_as_contextmanager(
    func, base_fn, expected_fn, expected_exc, ok_unless_exception
):
    """Test context manager usage."""
    with run_rbr_helper_contextmanager(
        base_fn, expected_fn, expected_exc=expected_exc
    ) as fn:
        with RenameByResult(fn, ok_unless_exception=ok_unless_exception) as rbr:
            rbr.set_result(func())


@pytest.mark.parametrize(
    "func, base_fn, expected_fn, expected_exc, ok_unless_exception", rbr_fixture
)
def test_rbr_as_decorator(
    func, base_fn, expected_fn, expected_exc, ok_unless_exception
):
    """Test decorator usage."""
    with run_rbr_helper_contextmanager(
        base_fn, expected_fn, expected_exc=expected_exc
    ) as fn:

        @RenameByResult(fn_arg="this_fn_arg", ok_unless_exception=ok_unless_exception)
        def f(this_fn_arg):
            assert this_fn_arg == fn
            return func()

        f(this_fn_arg=fn)


@pytest.mark.parametrize(
    "func, base_fn, expected_fn, expected_exc, ok_unless_exception", rbr_fixture
)
def test_rbr_as_decorator_inside_processpool(
    func, base_fn, expected_fn, expected_exc, ok_unless_exception
):
    """Test decorator usage inside process pool to ensure there won't be pickling or other
       issues.
    """
    with run_rbr_helper_contextmanager(
        base_fn, expected_fn, expected_exc=expected_exc
    ) as fn:

        @RenameByResult(fn_arg="this_fn_arg", ok_unless_exception=ok_unless_exception)
        def f(this_fn_arg):
            assert this_fn_arg == fn
            return func()

        with ThreadPoolExecutor(max_workers=5) as executor:
            fut = executor.submit(f, this_fn_arg=fn)

        fut.result()


def test_rbr_validation_init():
    with pytest.raises(ValueError):
        RenameByResult(fn="")
    with pytest.raises(ValueError):
        RenameByResult(fn_arg=None)


def test_rbr_validation_as_decorator():

    with pytest.raises(ValueError):

        @RenameByResult(fn_arg="this_isnt_fn_arg")
        def f(fn_arg):
            pass

        f()

    def f(fn_arg):
        pass

    rbr = RenameByResult(fn_arg="this_isnt_fn_arg")
    rbr.fn_arg = None
    with pytest.raises(ValueError):
        rbr(f)


def f_get_pid():
    """Just return current pid."""
    return os.getpid()


def run_poolexecutor_test(pool, is_multi_process=False, n_tasks=5):
    """Perform basic test pool excutor.

    Args:
        is_multi_process(int): Check if the pool indeed used multiple processes or no.
        n_tasks(int): Number tasks to run.
    """
    futures = []
    with pool as executor:
        for _ in range(n_tasks):
            futures.append(executor.submit(f_get_pid))

    wait(futures)
    pids = [f.result() for f in futures]
    assert n_tasks == len(pids)

    if is_multi_process:
        expected_num_pids = n_tasks
    else:
        expected_num_pids = 1

    assert len(set(pids)) == expected_num_pids


def test_DummyPoolExecutor():
    pool = DummyPoolExecutor()
    run_poolexecutor_test(pool, is_multi_process=False)


@pytest.mark.parametrize(
    "func, base_fn, expected_fn, expected_exc, ok_unless_exception",
    [(f_ok, f"{READY_PREFIX}_test", f"{DONE_PREFIX}_test", None, False),],
)
def test_process_new_files(
    func, base_fn, expected_fn, expected_exc, ok_unless_exception
):
    def proc_func(fn, rbr):
        pass

    with run_rbr_helper_contextmanager(
        base_fn, expected_fn, expected_exc=expected_exc
    ) as fn:
        in_dir = Path(fn).parent
        process_new_files(proc_func, in_dir)


def proc_func(fn):
    Path(fn).rename(str(fn).replace('ready_', 'done_'))


@pytest.mark.parametrize(
    "func, base_fn, expected_fn, expected_exc, ok_unless_exception, workers",
    [
        (f_ok, f"{READY_PREFIX}_test", f"{DONE_PREFIX}_test", None, False, None),
        (f_ok, f"{READY_PREFIX}_test", f"{DONE_PREFIX}_test", None, False, -1),
    ],
)
def test_process_new_files_parallel(
    func, base_fn, expected_fn, expected_exc, ok_unless_exception, workers
):

    with run_rbr_helper_contextmanager(
        base_fn, expected_fn, expected_exc=expected_exc
    ) as fn:
        # import pdb; pdb.set_trace()
        # in_dir = Path(fn).parent
        process_new_files_parallel(proc_func, [fn], workers=workers)
