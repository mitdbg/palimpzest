## Testing with Pytest in Palimpzest
- tests in `test_*.py` files
- fixtures in `conftest.py` and `fixtures/` (these are auto-discovered by `pytest` classes / tests)

### Running tests
Run the scripts within the `tests/pytest` directory:
- `cd tests/pytest` first
- `pytest` runs all tests
- `pytest -k <test_name>` runs a specific test
- `pytest -m <tag>` runs tests with a specific tag
