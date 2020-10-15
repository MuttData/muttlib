# Contributing to muttlib
Thanks for your interest in contributing to `muttlib` 🎉. These are the guidelines for contributions. Reading them will help you get started on how to make useful contributions.

## Foreword
This guide is not final. It will evolve over time, as we learn and add new voices to the project. Check it from time to time and feel free to make suggestions 😃

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Issues](#issues)
  - [Labels](#labels)
  - [Security issues](#security-issues)
- [Development Setup](#development-setup)
  - [Installation](#installation)
  - [Pre-Commit for Version Control Integration](#pre-commit-for-version-control-integration)
- [Style guide](#style-guide)
- [Docstrings](#docstrings)
- [Testing](#testing)
  - [Regression testing](#regression-testing)
- [Documentation](#documentation)
- [Versioning](#versioning)
- [Deprecation](#deprecation)
- [PRs](#prs)
  - [WIP](#wip)
  - [RFC](#rfc)
  - [CI/CD jobs](#cicd-jobs)
- [Rules of Thumb](#rules-of-thumb)

## Code of Conduct
One of our core values at Mutt is that **we are an open team**. We all make mistakes and need help fixing them. We foster psychological safety. We clearly express it when we don’t know something and ask for advice.

We expect everyone contributing to `muttlib` to follow this principle. Be kind, don't be rude, keep it friendly; learn, teach, ask, help.

## Issues

Before submitting an issue, first check on the [issues tracker](https://gitlab.com/mutt_data/muttlib/issues) if there is already one trying to cover that topic, to avoid duplicates. Otherwise we invite you to create it. And if you feel that your issue can be categorized you can use this labels:

### Labels

| name | description | shortcuts |
| ---------- | ----------- | ------ |
| `bug` | Report a bug | [Look](https://gitlab.com/mutt_data/muttlib/issues?scope=all&utf8=✓&state=opened&label_name[]=bug) for `bug` or [create](https://gitlab.com/mutt_data/muttlib/-/issues/new?issuable_template=Bug) one
|`feature-request`|Request for a new feature|[Look](https://gitlab.com/mutt_data/muttlib/issues?scope=all&utf8=✓&state=opened&label_name[]=feature-request) for `feature-request` or [create](https://gitlab.com/mutt_data/muttlib/-/issues/new?issuable_template=Feature) one
|`enhancement`|Propose an enhancement|[Look](https://gitlab.com/mutt_data/muttlib/issues?scope=all&utf8=✓&state=opened&label_name[]=enhancement) for `enhancement` or [create](https://gitlab.com/mutt_data/muttlib/-/issues/new?issuable_template=Enhancement) one
|`discussion`|Start a new discussion|[Look](https://gitlab.com/mutt_data/muttlib/issues?scope=all&utf8=✓&state=opened&label_name[]=discussion) for `discussion` or [create](https://gitlab.com/mutt_data/muttlib/-/issues/new?issuable_template=Discussion) one

### Security issues
If you find a security related bug or any kind of security rellated issue, **please DO NOT file a public issue**. Sensitive security-related issues should be reported to privately to the repo owner along with a PoC if possible. You can [send us an email](mailto:security@muttdata.ai) and we'll go from there.


## Development Setup

### Installation
Start by cloning the repo
```bash
git clone git@gitlab.com:mutt_data/muttlib.git
```

Then install all `dev` dependencies:
```bash
cd muttlib
pip install .[dev]
```

### Pre-Commit for Version Control Integration

We use [pre-commit](https://pre-commit.com) to run several code scans and hooks like linters and formatters, defined in `.pre-commit-config.yaml`, on each staged file  that make the development cycle easier.

To install pre-commit hooks run
```bash
pre-commit install
pre-commit install -t push
```

## Style guide
muttlib follows [PEP8](https://www.python.org/dev/peps/pep-0008/).

If you installed the [pre-commit hooks](#pre-commit) you shouldn't worry too much about style, since they will fix it for you or warn you about styling errors. We use the following hooks:

- [black](https://github.com/psf/black): An opinionated code formatting tool that ensures consistency across all projects using it
- [flake8](https://github.com/PyCQA/flake8): a tool to enforce style guide
- [mypy](https://github.com/python/mypy): a static type checker for Python
- [pylint](https://github.com/PyCQA/pylint): a source code, bug and quality checker

## Docstrings
We use either [numpy style](https://numpydoc.readthedocs.io/en/latest/format.html) or [google style](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings) docstring formatting. It's usually good to include the following docstrings:
- Module level docstring giving a general overview of what it does.
	- It may include TODOs
	- It may include examples
- Class dosctrings explaining what it is
- Method/functions to explain what it does and what it's parameters are

## Testing
We use the [pytest framework](https://docs.pytest.org/en/latest/) to test `muttlib`.

To run the default test suite run this:
```bash
pytest
```
Note: Some extra deps might be needed. Those can be added with this `pip install -e .[ipynb-utils]`.

Run coverage:
```bash
pytest --cov-report html:cov_html --tb=short -q --cov-report term-missing --cov=. tests/
```
That should output a short summary and generate a dir `cov_html/` with a detailed HTML report that can be viewed by opening `index.html` in your browser.

To run the tests with [nox](https://nox.thea.codes/en/stable/):
```bash
nox --session tests
```

### Regression testing
[Regression testing](https://en.wikipedia.org/wiki/Regression_testing) to ensure new changes have not broken previously working features.

## Documentation
`muttlib` uses [Sphinx](https://www.sphinx-doc.org/en/master/) to autogenerate it's [docs](https://mutt_data.gitlab.io/muttlib/) that are automatically built from [docstrings](#docstrings) and pushed by the [CI jobs](#cicd-jobs). Check the [style guide](#style-guide) section for notes on docstrings. Pushing all the docs is too cumbersome. You can generate them locally like so:

```bash
pip install .[dev]
cd docs
make html
```

And open `docs/build/html/index.html` on your browser of choice.

Alternatively you can see the docs for the `master` branch [here.](https://mutt_data.gitlab.io/muttlib/index.html)

## Versioning
We use [SemVer](https://semver.org). To keep things easy, we've included [bump](https://pypi.org/project/bump/) as a dev dependency. Running `bump` will bump the patch version. To bump minor/major versions:
```bash
bump --minor
bump --major
bump --patch
```

Please remember to bump the version when submitting your PR!

## Deprecation
Before fully deprecating a feature or making a breaking change, give users a warning and enough time for them to migrate their code. State when the EOL will be. Then, in the pertaining release, it can be included.

## PRs
Also called MRs (Merge Requests) in gitlab.

`muttlib` development follows a simple workflow:
- Assign yourself an issue
	- If there's none, [create it](#issues)
	- If you can't assign it yourself, ask someone to do it for you
- Create a new branch with a descriptive name
- Push to the remote
	- Open a [WIP](#WIP) PR to allow discussion and let others know where you're at with the issue
- Work on it 🤓
- When ready change the PR to [RFC](#RFC)
- You'll need at least one approval to merge
	- Merge will be disabled if the [CI/CD pipelines are failing](#cicd-jobs)
	- If you can't merge it yourself, ask your last approver to merge it
	- Please squash the commits and delete the branch
- Congrats and thanks for your contribution 🎉

Please keep PRs minimal. Try to keep the modified files to the bare needed for the issue you are working on. This will make the PR's changes more readable and allow for a quicker interaction with reviewers.

### WIP
WIP stands for **W**ork **i**n **P**rogress. WIP PRs are not yet ready to be merged. They allow for:
- Other project members to know you are working on something
- Early feedback, e.g. if you are doing something wrong or they see a problem down the road with your approach

You can tag a PR as WIP using the `WIP:` prefix on you PR title.

### RFC
RFC stands for **R**equest **f**or **C**omments. It means you consider the issue is solved by the code in the PR and are asking people to review your changes.

### CI/CD jobs

All commits pushed to branches in pull requests will trigger CI jobs that install `muttlib` in a gitlab-provided docker-env and all the extras, run all tests and check for linting. Look at [.gitlab-ci.yml](.gitlab-ci.yml) for more details on this and as well as the official [docs](https://docs.gitlab.com/ce/ci/README.html). Note that only PRs that pass the CI will be allowed to merge.

`NOTE:` If your commit message contains [ci skip] or [skip ci], without capitalization, the job will be skipped i.e. no CI job will be spawned for that push.

Alternatively, one can pass the ci.skip Git push option if using Git 2.10 or newer: `git push -o ci.skip` more info in [here](https://docs.gitlab.com/ce/ci/yaml/README.html#skipping-builds).

`IMPORTANT:`. If you skip the CI job it will not disable the option to do the merge, be careful when doing this.

**Important note on coverage:** A regex that captures the output from `pytest-cov` has been set from Settings -> CI/CD -> General Pipelines -> Test coverage parsing

## Rules of Thumb
- Important changes should be mentioned in the [README.md](README.md)
- Documentation must be updated.
- Every change should be present in the [CHANGELOG.md](CHANGELOG.md)
