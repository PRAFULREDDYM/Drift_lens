# Contributing to drift-lens

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
# 1. Fork the repo on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/drift-lens.git
cd drift-lens

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 3. Install in editable mode with dev dependencies
pip install -e ".[dev]"

# 4. Verify everything works
pytest tests/ -v
```

## Making Changes

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** — keep commits focused and descriptive.

3. **Add tests** for any new functionality in `tests/`.

4. **Run the test suite** to make sure nothing is broken:
   ```bash
   pytest tests/ -v
   ```

5. **Submit a pull request** against `main`.

## What We're Looking For

- 🐛 **Bug fixes** — always welcome
- 📝 **Documentation** — better examples, typos, clarity
- 🧪 **Tests** — more coverage is always good
- 🚀 **New detection methods** — implement the `compare()` interface in `drift_lens/detector/`
- 🎨 **Dashboard improvements** — Streamlit UI enhancements
- 📦 **Integrations** — adapters for MLflow, W&B, etc.

## Code Style

- We use standard Python conventions (PEP 8)
- Type hints are required for all public functions
- Docstrings follow NumPy style
- All magic numbers live in `drift_lens/constants.py` with explanations

## Adding a New Detection Method

1. Create `drift_lens/detector/your_method.py`
2. Implement:
   ```python
   def your_method_compare(
       baseline: np.ndarray, current: np.ndarray
   ) -> tuple[float, float | None, dict[str, Any]]:
       # Return (normalized_score, p_value_or_none, details_dict)
   ```
3. Register it in `drift_lens/detector/__init__.py` → `_METHOD_DISPATCH`
4. Add CLI support in `drift_lens/cli.py` → method `Choice` lists
5. Add tests in `tests/test_detector.py`

## Reporting Issues

- Use [GitHub Issues](https://github.com/PRAFULREDDYM/drift-lens/issues)
- Include: Python version, OS, drift-lens version, minimal reproduction steps

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
