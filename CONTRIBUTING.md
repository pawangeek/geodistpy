# Contributing to Geodistpy

Thank you for your interest in contributing to **geodistpy**! We welcome contributions of all kinds — bug reports, feature requests, documentation improvements, and code contributions.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Reporting Bugs](#reporting-bugs)
- [Requesting Features](#requesting-features)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to **pawanjain.432@gmail.com**.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone git@github.com:<your-username>/geodistpy.git
   cd geodistpy
   ```
3. **Add the upstream remote:**
   ```bash
   git remote add upstream git@github.com:pawangeek/geodistpy.git
   ```

## Development Setup

Geodistpy uses [Poetry](https://python-poetry.org/) for dependency management. Make sure you have Poetry installed, then:

```bash
# Install all dependencies (including dev/test/lint groups)
poetry install --with test,lint

# Activate the virtual environment
poetry shell
```

### Requirements

- Python >= 3.11
- [Numba](https://numba.pydata.org/) (JIT compilation)
- [NumPy](https://numpy.org/)
- [geographiclib](https://geographiclib.sourceforge.io/Python/) (fallback for edge cases)

## Making Changes

1. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/my-feature
   # or
   git checkout -b fix/my-bugfix
   ```

2. **Make your changes.** Keep commits focused and atomic.

3. **Write or update tests** for any new functionality or bug fixes.

4. **Update documentation** if your changes affect the public API. Documentation lives in:
   - `docs/api-reference.md` — Function reference
   - `docs/getting-started.md` — Quick start guide
   - `docs/explanation.md` — Benchmarks and implementation details
   - `README.md` — Project overview

## Testing

We use [pytest](https://docs.pytest.org/) for testing. Run the full test suite with:

```bash
# Run all tests
poetry run pytest

# Run with verbose output
poetry run pytest -v

# Run a specific test
poetry run pytest tests/test_geodist.py::test_bearing_due_east

# Run with coverage
poetry run pytest --cov=geodistpy
```

**All tests must pass before submitting a pull request.** If you add a new feature, please include comprehensive tests covering:

- Normal/expected behavior
- Edge cases (coincident points, poles, antimeridian, etc.)
- Invalid input (out-of-range coordinates, wrong shapes)
- Metric unit conversions

## Code Style

We use [Black](https://black.readthedocs.io/) for code formatting:

```bash
# Format code
poetry run black geodistpy/ tests/

# Check without modifying
poetry run black --check geodistpy/ tests/
```

We also use [codespell](https://github.com/codespell-project/codespell) for spell checking:

```bash
poetry run codespell
```

### Style Guidelines

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions
- Use type hints where practical
- Write clear docstrings for all public functions (Google/NumPy style)
- Keep Numba-JIT functions (`@jit(nopython=True)`) in `geodesic.py` and high-level Python wrappers in `distance.py`
- Include `Examples` sections in docstrings

## Submitting a Pull Request

1. **Push** your branch to your fork:
   ```bash
   git push origin feature/my-feature
   ```

2. **Open a Pull Request** against the `main` branch of `pawangeek/geodistpy`.

3. In your PR description, please include:
   - A clear summary of the changes
   - The motivation / use case
   - Any relevant issue numbers (e.g., "Fixes #42")

4. **Wait for CI checks** to pass (lint, tests, build).

5. A maintainer will review your PR. Please be responsive to feedback.

### PR Checklist

- [ ] All existing tests pass (`poetry run pytest`)
- [ ] New tests added for new functionality
- [ ] Code formatted with Black
- [ ] Documentation updated (if applicable)
- [ ] Commit messages are clear and descriptive

## Reporting Bugs

If you find a bug, please [open an issue](https://github.com/pawangeek/geodistpy/issues/new) with:

- A clear, descriptive title
- Steps to reproduce the bug
- Expected vs. actual behavior
- Your environment (Python version, OS, geodistpy version)
- A minimal code example that demonstrates the issue

## Requesting Features

Feature requests are welcome! Please [open an issue](https://github.com/pawangeek/geodistpy/issues/new) with:

- A clear description of the proposed feature
- The use case / motivation
- Example API usage (how you'd like to call it)

## License

By contributing to geodistpy, you agree that your contributions will be licensed under the [MIT License](LICENSE).

---

Thank you for helping make geodistpy better! 🌍
