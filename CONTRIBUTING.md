# Contribution guide

Welcome! We appreciate your interest in contributing to Tunix. This guide
details how to contribute to the project in a way that is efficient for
everyone.

We follow
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).

## Contributing code

### 1. Propose Changes in an Issue

Before starting on your contribution, please check for an existing issue.

For significant changes, please open an issue to discuss your proposal first.
This allows the team to provide feedback and ensure the change aligns with the
project's goals.

For minor changes, such as documentation updates or simple bug fixes, you can
open a pull request directly.

All bug fixes must include a link to a
[Colab](https://colab.research.google.com/) notebook that clearly reproduces
the error.

### 2. Make code changes

To begin coding, fork the repository and create a new branch from main.

### 3. Create a pull request

Once your changes are ready, open a pull request from your branch to the main
branch of the upstream Tunix repository. Please provide a clear title and
description, linking to the relevant issue if one exists.

### Step 4. Sign the Contributor License Agreement

If this is your first contribution, you will be prompted to sign the Google CLA
after submitting your pull request. You can review the agreement
[here](https://cla.developers.google.com/clas).

### Step 5. Code review

A project maintainer will review your pull request. Be prepared for one or more
rounds of comments and requested changes as we work with you to refine the
contribution.

### Step 6. Merging

Once the pull request is approved, a team member will take care of merging.
Thank you for your contribution!

## Formatting and linting

We use [Pyink](https://github.com/google/pyink) and
[Pylint](https://github.com/pylint-dev/pylint) for formatting and linting,
respectively.

For the first time you are setting up the repo, please run `pre-commit install`.
Note that this needs to be done only once at the beginning.

Now, you go through the usual flow of pushing code:

```
git add .
git commit -m "<message>"
```

Whenever you run `git commit -m "<message>"`, the code is automatically
formatted, and lint error messages are displayed.

If there's any error, the commit will not go through. Most of the times, the
errors are fixed automatically.

Note: Pylint errors are not binding, i.e., your commit will not fail if you have
linting errors. This is because Pylint is very strict. It is, however, advised
that you address most of the errors.

Once you are done fixing your errors, re-run the following:

```
git add .
git commit -m "<message>" # This will not get logged as a duplicate commit.
```

In case you want to run the above manually on all files, you can do the
following:

```
pre-commit run --all-files
```

If you want to opt out of pre-commit, you can always do the following (but make
sure you run the pre-commit hooks manually):

```
git commit -m "<message>" --no-verify
```
