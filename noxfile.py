"""Entrypoint for nox."""
import nox


@nox.session(reuse_venv=True)
def tests(session):
    """Run all tests."""
    session.install(".[all]")
    cmd = ["pytest", "-n", "auto"]
    if session.posargs:
        cmd.extend(session.posargs)
    session.run(*cmd)


@nox.session(reuse_venv=True)
def cop(session):
    """Run all pre-commit hooks."""
    session.install(".[all]")
    session.run("pre-commit", "install")
    session.run("pre-commit", "run", "--show-diff-on-failure", "--all-files")


@nox.session(reuse_venv=True)
def bandit(session):
    """Run all pre-commit hooks."""
    session.install("bandit")
    session.run("bandit", "-r", "muttlib/", "-ll", "-c", "bandit.yaml")
