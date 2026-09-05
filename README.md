# inference-ladder

Climbing the inference ladder on one mechanistic model.

A single SIR/COVID model is fitted again and again, each time with a method one step
stronger than the last — from a log-linear slope to a full posterior — and every step
has to earn its place by answering a question the previous one could not. On top of that
foundation, three distinct ways of inserting machine learning into a mechanistic model are
compared against it.

> **Status:** scaffolding. The ladder itself is not built yet.

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Development

```bash
uv run ruff check .        # lint
uv run ruff format .       # format
uv run pytest              # tests
uv run pre-commit install  # enable hooks
```

## Layout

```
src/inferladder/   # the package
tests/             # pytest, mirrors src/
notebooks/         # one notebook per rung
report/            # narrative artifact
data/              # gitignored, see data/README.md
```

## License

MIT — see [LICENSE](LICENSE).
