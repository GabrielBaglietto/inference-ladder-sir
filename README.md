# The inference ladder on SIR

Six rungs on the same mechanistic model, in increasing order of cost, where each one is on
the ladder because it answers a question the one below it could not — and each one closes
by saying what it settles, what it does not, and when you would be right to stop there.

The model is a deterministic SIR with an explicit observation model (under-reporting,
Poisson or negative-binomial counts). The data are synthetic with known ground truth, so
every rung can be checked against the answer instead of argued about.

> **Status:** complete as a study of rungs 0–5. No further rungs are planned here.

## What the climb found

Three results, and they are the same geometry seen from three rungs.

**Over a complete epidemic the likelihood is well behaved, and that is not the good news it
sounds like.** It has a single optimum, and twelve independent searches from different
starting points all find it. What they also find is a long flat valley — and a flat valley
is not an optimiser's problem. It is a question about whether the data determine the
parameters at all.

**Whether they do depends on the observation window, which is the opposite of the usual
shorthand.** Over a complete epidemic both rates and `R0` are pinned down. Over the
exponential phase alone only `beta - gamma` is, and `R0` is unbounded — so the quantity
normally treated as the rigid one is precisely the one early data cannot fix.

**The valley is also what breaks samplers that ignore it.** `beta` and `gamma` correlate at
0.99 in the posterior. A componentwise Metropolis retains an effective sample size in the
single digits out of 3200 draws; a sampler that learns proposal directions from the chain's
own history gets around 570, in a quarter of the runtime. And since a Laplace approximation
already agrees with the posterior on the marginals, the reason to sample at all is the joint
distribution and the priors — not tighter intervals.

## The rungs

| | question | method |
|---|---|---|
| `00` | how fast is it growing? | log-linear fit on the exponential phase |
| `01` | what are the rates separately? | least squares vs. maximum likelihood |
| `02` | is that really the optimum? | multistart, cost surfaces |
| `03` | do the data determine the parameters? | sensitivity spectrum, observed Fisher, profile likelihood |
| `04` | how uncertain is the fit? | Laplace, profile and bootstrap intervals |
| `05` | what can we decide? | posterior sampling, stated priors, a decision curve |

Rung 5 is not NUTS. NUTS needs the gradient of the log-posterior, and the dynamics are
integrated by `scipy`, which automatic differentiation cannot see through. Supplying
gradients would mean a second implementation of the same model in a differentiable
framework — the drift the model-agnostic design exists to prevent. The sampler is
`DEMetropolisZ`, and notebook 05 measures what that choice costs.

## On prior art

Implementations of these methods on SIR already exist, and `sir-julia` is the broadest of
them: it covers more formalisms than this repository does, and anyone comparing the two on
coverage should prefer it.

What a catalogue of recipes cannot have is the chain — each method appearing because the
previous one failed for a stated reason, the failures reported rather than smoothed over,
and every rung declaring its own stopping criterion. That chain is what this repository
is, and it is the only thing it claims.

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
data/              # gitignored, see data/README.md
```

The package is split so that no rung knows which model it is running on:
`models/base` defines the `SystemModel` / `ObservationModel` split and the
`GenerativeModel` that composes them, and every inference module takes that interface.

## Running the notebooks

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
small. Executing the full set takes about twelve minutes on a laptop, nearly all of it the
bootstrap in rung 4 and the posterior sampling in rung 5. Seeds are fixed throughout: the
numbers in the text are the numbers you get.

## License

MIT — see [LICENSE](LICENSE).
