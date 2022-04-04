"""Entrypoint for nox."""
import nox_poetry as nox


@nox.session(reuse_venv=True, python=["3.8", "3.9"])
def tests(session):
    """Run all tests."""
    session.run_always("poetry", "install", "-E", "all", "-vv", external=True)
    cmd = ["poetry", "run", "pytest", "-n", "auto", "--mpl"]
    if session.posargs:
        cmd.extend(session.posargs)
    session.run(*cmd)


@nox.session(reuse_venv=True, python=["3.8", "3.9"])
def precommit_hooks(session):
    """Run all pre-commit hooks."""
    session.run_always("poetry", "install", "-E", "all", "-vv", external=True)
    session.run("pre-commit", "install")
    session.run("pre-commit", "run", "--show-diff-on-failure", "--all-files")


@nox.session(reuse_venv=True, python=["3.8", "3.9"])
def bandit(session):
    """Run all pre-commit hooks."""
    session.run_always("poetry", "install", "-E", "all", "-vv", external=True)
    session.run("bandit", "-r", "muttlib/", "-ll", "-c", "bandit.yaml")
