# Contributing to Nesion

First off, thank you for considering contributing to Nesion. People like you make Nesion such a great tool for the ML inference community.

### 1. Where do I go from here?

If you've noticed a bug or have a feature request, make one! It's generally best if you get confirmation of your bug or approval for your feature request this way before starting to code.

### 2. Fork & create a branch

If this is something you think you can fix, then fork Nesion and create a branch with a descriptive name.

### 3. Get the test suite running

Make sure you're using a virtual environment (like `venv` or `conda`). Then explicitly download the source requirements using our Makefile.

```bash
make install
```

### 4. Code Standards

Nesion mandates `black` and `ruff` formatting standards on all Python code bases. Before committing your branches to PR, ensure you:

```bash
make format
make lint
```

Then invoke tests:

```bash
make test
```

### 5. Pull Requests
Once you're ready, submit a PR! The CI/CD pipelines will automatically check your work through our GitHub Actions `test` and `lint` matrices (Python `3.10`, `3.11` and `3.12` tested concurrently).
