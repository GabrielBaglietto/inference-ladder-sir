# inference-ladder

Climbing the inference ladder on one mechanistic model.

A single SIR/COVID model is fitted again and again, each time with a method one step
stronger than the last — from a log-linear slope to a full posterior — and every step
has to earn its place by answering a question the previous one could not. On top of that
foundation, three distinct ways of inserting machine learning into a mechanistic model are
compared against it.

> **Status:** the mechanistic ladder is complete, rungs 0–5 — from a log-linear growth
> rate to a joint posterior. The machine-learning insertions are next.

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

## The ladder so far

Each rung answers a question the one below it could not, and each notebook closes by
saying what it settles and when you would be right to stop there.

| | question | method |
|---|---|---|
| `00` | how fast is it growing? | log-linear fit on the exponential phase |
| `01` | what are the rates separately? | least squares vs. maximum likelihood |
| `02` | is that really the optimum? | multistart, cost surface |
| `03` | do the data determine the parameters? | sensitivity spectrum, profile likelihood |
| `04` | how uncertain is the fit? | Laplace, profile and bootstrap intervals |
| `05` | what can we decide? | posterior sampling, priors, a decision curve |

Notebooks are stored **without outputs**, so they arrive as empty cells and have to be
run. Three ways, depending on what you want:

```bash
# execute all of them and check they still work (no output kept)
uv sync --group bayes          # notebook 05 samples a posterior
uv run pytest --nbmake notebooks/

# render them to standalone HTML you can open in a browser
uv sync --group notebooks
uv run jupyter nbconvert --execute --to html notebooks/*.ipynb --output-dir notebooks/_rendered

# work on them interactively
uv sync --group notebooks
uv run jupyter lab
```

JupyterLab lives in its own `notebooks` dependency group, so a plain `uv sync` stays
small and CI never installs it. Executing the full set takes roughly three minutes,
most of it the bootstrap in rung 4.

## License

MIT — see [LICENSE](LICENSE).
