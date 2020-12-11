"""Entrypoint for nox."""
import nox


@nox.session(reuse_venv=True, python=["3.7", "3.8"])
def tests(session):
    """Run all tests."""
    session.install("pip==20.3.1")
    session.install(".[all]", "--use-deprecated=legacy-resolver")
    cmd = ["pytest", "-n", "auto"]
    if session.posargs:
        cmd.extend(session.posargs)
    session.run(*cmd)


@nox.session(reuse_venv=True, python=["3.7", "3.8"])
def cop(session):
    """Run all pre-commit hooks."""
    session.install("pip==20.3.1")
    session.install(".[all]", "--use-deprecated=legacy-resolver")
    session.run("pre-commit", "install")
    session.run("pre-commit", "run", "--show-diff-on-failure", "--all-files")


@nox.session(reuse_venv=True, python=["3.7", "3.8"])
def bandit(session):
    """Run all pre-commit hooks."""
    session.install("pip==20.3.1")
    session.install("bandit", "--use-deprecated=legacy-resolver")
    session.run("bandit", "-r", "muttlib/", "-ll", "-c", "bandit.yaml")
