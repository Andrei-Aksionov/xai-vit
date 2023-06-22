<p>
    <h1 align="center">eXplainable AI</h1>
    <h6 align="center">for</h6>
    <h3 align="center">Vision Transformers</h3>
</p>

![Python versions](/assets/readme/python_versions.svg)

TODO: change link
[![test](https://github.com/Andrei-Aksionov/nanoGPTplus/actions/workflows/test.yaml/badge.svg)](https://github.com/Andrei-Aksionov/nanoGPTplus/actions/workflows/test.yaml)

***

<p align=center><img src="assets/readme/nn_scheme.png"></p>

> **Note**: This is a long going project. That means that the state of the repo is far from being finished: when I find something interesting for ViT interoperability - I'll implement it here.

I have been planning for quite a while to work on a project that tries to uncover hidden nuances of how transformer architecture works for computer vision tasks such as:

- classification
- object detection
- segmentation

In this repository the subject matter is Vision Transformer (ViT) - an adaptation of the encoder part of transformer architecture for computer vision tasks.

ROADMAP:

- [x] Implement ViT for image classification from scratch. That will help analyze process of "thinking" later on.
- [] Create "heatmap" for attentions. That should help to visualize on which parts of image the model pays more attention during forward pass.
- [] Analyze heads for multi-head attention. Some of them should focus on short-distance tokens, while other - on long-distance ones. I would like to check whether it's true or not.
- [] Repeat the same for the task of segmentation.
- [] Repeat the same for the task of object detection.
- [] And beyond ...

# How to use it

1. Install all the required packages. In this project all the python dependencies are managed by [Poetry](https://python-poetry.org/) and are stored in `pyproject.toml` file alongside required version of python. After `poetry` is installed and virtual environment is created (in case you don't want poetry to create it [automatically](https://python-poetry.org/docs/configuration/#virtualenvscreate)), run:

    ```bash
    poetry install
    ```

2. Check notebooks from `notebooks` folder.

    All insights will be stored there.

3. If you want to experiment with the implementation - run tests to make sure that everything works as expected.

## Run tests

[Pytest](https://github.com/pytest-dev/pytest) framework is used for tests execution so in order to run all tests simply type:

```bash
pytest
```

> **Note**: the command above will run all the tests.

There are two types of tests: fast and slow ones. Slow tests are marked with `slow` marker. If one wants to run only fast test:

```bash
pytest -m "not slow"
```

If one wants to see also standard output in pytest logs:

```bash
pytest --include=sys
```

***

## Additional: pre-commit hooks

In order to install pre-commit hooks run:

```bash
pre-commit install
```

Pre-commit hooks are be executed before each commit. In addition all the pre-commit hooks will be run per each PR via github-workflow (no need to add or change anything).

The list of all hooks one can find in a config fils: `.pre-commit-config.yaml`

**Note**: for the sake of speed pre-commit hooks will be executed only on changed files. If it's needed to run on all files execute:

```bash
pre-commit run --all-files
```
