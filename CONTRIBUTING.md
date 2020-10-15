# Contributing to muttlib
Thanks for your interest in contributing to muttlib 🎉 These are the guidelines for contributions. Reading them will help you get started on how to make useful contributions.

## Foreword
This guide is not final. It will evolve over time, as we learn and add new voices to the project. Check it from time to time and feel free to make suggestions 😃

## Code of conduct
One of our core values at Mutt is that **we are an open team**. We all make mistakes and need help fixing them. We foster psychological safety. We clearly express it when we don’t know something and ask for advice.

We expect everyone contributing to muttlib to follow this principle. Be kind, don't be rude, keep it friendly; learn, teach, ask, help.

## Issues

### Feature requests
To ask for new functionalities or improvements on muttlib, please head over to the [issues tracker](https://gitlab.com/mutt_data/muttlib/issues) and open a new issue with the label `feature-request` attached to it or directly use our [feature request](https://gitlab.com/mutt_data/muttlib/-/issues/new?issuable_template=Feature) template.

A brief description of the desired functionality should be included; use cases and examples are welcome too, for a better understanding of the request.

Feel free to [open a new MR](#mrs) for it!

### Bug reporting
To report a bug, please check there is no previous related issue on the [issues tracker](https://gitlab.com/mutt_data/muttlib/issues) and then open a new issue with the `bug` label attached to it or directly use our [report a bug](https://gitlab.com/mutt_data/muttlib/-/issues/new?issuable_template=Bug) template.

A MWE example of the bug should be provided to allow the person working on fixing it to save time reproducing and debugging it. Also, it should act as a test to check if whichever changes are made do effectively fix it. [MRs are welcome](#mrs) to fix them!

#### Security issues
If you find a security related bug or any kind of security rellated issue, **please DO NOT file a public issue**. Sensitive security-related issues should be reported to privately to the repo owner along with a PoC if possible. You can [send us an email](mailto:security@muttdata.ai) and we'll go from there.



## Contributing

### Dev install
Start by cloning the repo
```bash
git clone git@gitlab.com:mutt_data/muttlib.git
```

Then install all `dev` dependencies:
```bash
cd muttlib
pip install .[dev]
```

#### pre-commit
We use [pre-commit](https://pre-commit.com) to run several code scans and hooks that make the development cycle easier. To install pre-commit hooks run
```bash
pre-commit install
pre-commit install -t push
```

### Development Setup

#### Pre-Commit for Version Control Integration

When developing you'll have to use the python library
[pre-commit](https://pre-commit.com/) to run a series of linters and formatters, defined
in `.pre-commit-config.yaml`, on each staged file.  There are two ways to install these
binaries:

#### Global install of binaries

The easiest way to set this up is by first installing `pipx` with

```bash
pip3 install --user pipx
pipx ensurepath
```

and then use `pipx` to actually install the `pre-commit` binary along the linters and
formatters globally:

```bash
pipx install pre-commit --verbose
pipx install flake8 --spec git+https://github.com/PyCQA/flake8 --verbose
pipx inject flake8 flake8-bugbear flake8-docstrings --verbose
pipx install black --verbose
pipx install mypy --verbose
pipx install pylint --verbose
```

Once that's done, `cd` into the repo where `.pre-commit-config.yaml` exists, run
`pre-commit install` and you are good to go: every time you do a `git commit` it will run
the `pre-commit` hooks defined in `.pre-commit-config.yaml`. You can also install a pre-push hook with `pre-commit install -t pre-push`, which will additionally run all pytest tests before pushing.

#### Local install of binaries

The binaries are also listed as `dev` packages in `setup.py`. Therefore you can
alternatively install `muttlib` locally in a virtual environment using `pipenv`. To do
that first clone the repo, `cd` into this `muttlib` folder and then run

```bash
pipenv install -e .[dev] --skip-lock
```

Since the `.pre-commit-config.yaml` forces `pre-commit` to execute the environment of the
shell at the time of `git commit` you'll then have to run `git commit` from within a
`pipenv` subshell by first running `pipenv shell`.

### Style guide
muttlib follows [PEP8](https://www.python.org/dev/peps/pep-0008/).

If you installed the [pre-commit hooks](#pre-commit) you shouldn't worry too much about style, since they will fix it for you or warn you about styling errors. We use the following hooks:

- [black](https://github.com/psf/black): An opinionated code formatting tool that ensures consistency across all projects using it
- [flake8](https://github.com/PyCQA/flake8): a tool to enforce style guide
- [mypy](https://github.com/python/mypy): a static type checker for Python
- [pylint](https://github.com/PyCQA/pylint): a source code, bug and quality checker

#### Docstrings
We use either [numpy style](https://numpydoc.readthedocs.io/en/latest/format.html) or [google style](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings) docstring formatting. It's usually good to include the following docstrings:
- module level docstring giving a general overview of what it does.
	- it may include TODOs
	- it may include examples
- class dosctrings explaining what it is
- method/functions to explain what it does and what it's parameters are

### Testing
We use the [pytest framework](https://docs.pytest.org/en/latest/) to test muttlib. Check the [readme](https://gitlab.com/mutt_data/muttlib#testing) on how to run tests.

### Docs
We uses [Sphinx](https://www.sphinx-doc.org/en/master/) to autogenerate it's [docs](https://mutt_data.gitlab.io/muttlib/) that are automatically built from [docstrings](#docstrings). Check the [style guide](#style-guide) section for notes on docstrings. Pushing all the docs is too cumbersome. You can generate them locally like so:

```bash
pip install .[dev]
cd docs
make html
```

And open `docs/build/html/index.html` on your browser of choice.

Alternatively you can see the docs for the `master` branch [here.](https://mutt_data.gitlab.io/muttlib/index.html)

### Versioning
We use [SemVer](https://semver.org). To keep things easy, we've included [bump](https://pypi.org/project/bump/) as a dev dependency. Running `bump` will bump the patch version. To bump minor/major versions:
```bash
bump --minor
bump --major
bump --patch
```

Please remember to bump the version when submitting your MR!

### Deprecation
Before fully deprecating a feature or making a breaking change, give users a warning and enough time for them to migrate their code. State when the EOL will be. Then, in the pertaining release, it can be included.

### MRs
muttlib development follows a simple workflow:
- assign yourself an issue
	- if there's none, [create it](#issues)
	- if you can't assign it yourself, ask someone to do it for you
- create a new branch with a descriptive name
- push to the remote
	- open a [WIP](#WIP) MR to allow discussion and let others know where you're at with the issue
- work on it 🤓
- when ready change the MR to [RFC](#RFC)
- you'll need at least one approval to merge
	- merge will be disabled if the [CI/CD pipelines are failing](#cicd-jobs)
	- if you can't merge it yourself, ask your last approver to merge it
	- please squash the commits and delete the branch
- congrats and thanks for your contribution 🎉

Please keep MRs minimal. Try to keep the modified files to the bare needed for the issue you are working on. This will make the MR's changes more readable and allow for a quicker interaction with reviewers.

#### WIP
WIP stands for **W**ork **i**n **P**rogress. WIP MRs are not yet ready to be merged. They allow for:
- other project members to know you are working on something
- early feedback, e.g. if you are doing something wrong or they see a problem down the road with your approach

You can tag a MR as WIP using the `WIP:` prefix on you MR title.

#### RFC
RFC stands for **R**equest **f**or **C**omments. It means you consider the issue is solved by the code in the MR and are asking people to review your changes.

### CI/CD jobs
By default, when creating new merge-requests to this lib, gitlab will spawn a
CI job for every push done to the remote branch. This will install muttlib in a
gitlab-provided docker-env that'll install all the extras specified in the
`.gitlab-ci.yml` file.

Then, the CI jobs will run a `setup.py test` for every push. If one test fails
then that merge request's merge option will be disabled.
**Note**: If your commit message contains [ci skip] or [skip ci], without
capitalization, the job will be skipped i.e. no CI job will be spawned for that
push.

> [Regression testing](https://en.wikipedia.org/wiki/Regression_testing) to ensure new changes have not broken previously working features.

Alternatively, one can pass the ci.skip Git push option if using Git 2.10 or newer: `git push -o ci.skip`
more info in https://docs.gitlab.com/ce/ci/yaml/README.html#skipping-builds.
`IMPORTANT`. If you skip the CI job it will not disable the option to do the merge, be careful when doing this.

**Important note on coverage:** A regex that captures the otuput from `pytest-cov` has been set from Settings -> CI/CD -> General Pipelines -> Test coverage parsing

**Important note**: the service-account credentials will effectively provide us with a google-valid email, which will act as the "user" editing/modifying/etc. the data in Google Sheets. 
This implies that this service email needs to have sufficient permissions to actually edit these files. 
In general, giving permissions to the needed sheets will suffice.

### README
Relevant changes (e.g. new modules) should be included in the README.

## Misc

### Issue labels

| label name | description | issues |
| ---------- | ----------- | ------ |
| bug | Bugs report or suspected bugs | [search](https://gitlab.com/mutt_data/muttlib/issues?scope=all&utf8=✓&state=opened&label_name[]=bug)|
|feature-request|Request for new features to add to muttlib|[search](https://gitlab.com/mutt_data/muttlib/issues?scope=all&utf8=✓&state=opened&label_name[]=feature-request)|

