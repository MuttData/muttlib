"""Entrypoint for nox."""
import nox


@nox.session(reuse_venv=True, python=["3.7", "3.8"])
def tests(session):
    """Run all tests."""
    session.env.update({"POETRY_VIRTUALENVS_CREATE": "false"})
    session.install("poetry")
    session.run("poetry", "install", "-E", "all")
    cmd = ["poetry", "run", "pytest", "-n", "auto", "--mpl"]
    if session.posargs:
        cmd.extend(session.posargs)
    session.run(*cmd)


@nox.session(reuse_venv=True, python=["3.7", "3.8"])
def precommit_hooks(session):
    """Run all pre-commit hooks."""
    session.env.update({"POETRY_VIRTUALENVS_CREATE": "false"})
    session.install("poetry")
    session.run("poetry", "install", "-E", "all")
    session.run("poetry", "run", "pre-commit", "install")
    session.run(
        "poetry", "run", "pre-commit", "run", "--show-diff-on-failure", "--all-files"
    )


@nox.session(reuse_venv=True, python=["3.7", "3.8"])
def bandit(session):
    """Run all pre-commit hooks."""
    session.env.update({"POETRY_VIRTUALENVS_CREATE": "false"})
    session.install("poetry")
    session.run("poetry", "install")
    session.run("poetry", "run", "pip", "install", "bandit==1.7.1")
    session.run("poetry", "run", "bandit", "-r", "muttlib/", "-ll", "-c", "bandit.yaml")
