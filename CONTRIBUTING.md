# Contributing to Dendra Models

Thank you for helping improve the model library. Public contributions are made
through GitHub pull requests.

## Set up a development checkout

Fork `wmglab-duke/dendra-models` on GitHub, clone your fork, and create a branch
for the change. From the repository root, create an isolated Python 3.11 or
later environment and install the project in editable mode:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --editable ".[dev]"
```

On Windows, activate the environment with `.venv\Scripts\activate`.
The standard installation includes Dendra's native CPU solvers.

## Make and test a change

Keep each pull request focused. Add a regression test when changing numerical
behavior, initialization, model parameters, or packaged resources. Run the
public test suite from the repository root:

```sh
python -m pytest -q \
  tests \
  src/dendra_models/models/networks/zhang_2014/tests
```

For a new model or parameterization, include:

- the source publication and the origin of any parameter or morphology data;
- the implemented units, default protocol assumptions, and initialization;
- a focused check of a scientifically meaningful reference behavior; and
- documentation showing how users construct the model.

Do not commit credentials, local filesystem paths, generated caches,
manuscript artifacts, or large simulation outputs. Discuss new large datasets
in an issue before adding them so they can be distributed and cited properly.

## Submit the pull request

Push your branch to your fork and open a pull request against
`wmglab-duke/dendra-models:main`. Describe the motivation, implementation, and
tests you ran. Maintainers may ask for a smaller reproducer or additional
scientific validation before merging a model change.

Except for the separately licensed cortical morphology files identified in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), by contributing you agree that
your contribution may be distributed under the Duke terms in
[LICENSE.md](LICENSE.md).

Changes to the cortical `.gml` morphology files are adapted material under
[CC BY-NC-SA 4.0](LICENSES/CC-BY-NC-SA-4.0.txt). Contributions to those files
must preserve their attribution and license, identify the modifications made,
and be distributable under the same Creative Commons terms. Only submit data or
other third-party material when you have the right to redistribute it and have
documented its source and license.
