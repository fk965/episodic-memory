# Contributing

We welcome contributions from the community. Here's how to get started.

## Development Setup

```bash
git clone https://github.com/fk965/episodic-memory.git
cd episodic-memory

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
make install

# Verify setup
make test
```

## Pull Request Process

1. Fork the repo and create a feature branch
2. Add tests for any new functionality
3. Ensure all tests pass: `make test`
4. Run lint and type checking: `make lint && make typecheck`
5. Open a PR with a clear title and description

### PR Title Convention

```
feat: add support for custom embedding models
fix: handle empty memory store gracefully
docs: clarify verification API behavior
```

## Code Standards

- Python 3.10+ with type hints
- Follow ruff formatting rules
- Test coverage >= 90% for new code
- One logical change per commit

## Issues

- Bug reports: include Python version, OS, and a minimal reproduction
- Feature requests: describe your use case, not just a solution

## Code of Conduct

All participants are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md).