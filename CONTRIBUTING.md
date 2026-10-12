# Contributing

Thank you for helping. Small, focused pull requests are easiest to review.

## Before you start

- For a bug, open an issue with the smallest code that shows it, the model name, and the full error.
- For a new feature, open an issue first. This SDK stays small on purpose: it speaks two request formats, and anything that needs provider translation belongs in a gateway.

## Making a change

1. Set up as in [DEVELOPMENT.md](DEVELOPMENT.md).
2. Add a test against the fake provider in `tests/fake_provider.py`.
3. Run `pytest`, `ruff check .` and `ruff format --check .`.
4. If you changed the public API on purpose, run `python scripts/surface.py --write` and explain why in the pull request.
5. Add a line to `CHANGELOG.md` under "Unreleased" if users will notice the change.

## Adding a provider

A provider is one row in `local_gateway.PROVIDERS`: the `aigw` schema it speaks, its address, and the environment variables for its key. The SDK never calls it directly; the local gateway does. Add a test, a row in the providers table in `docs/usage.md`, and a line in the changelog.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).

## License

By contributing, you agree that your contribution is licensed under the [Apache License 2.0](LICENSE).
