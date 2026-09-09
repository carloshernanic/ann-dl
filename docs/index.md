# Artificial Neural Networks & Deep Learning

???+ info inline end "Edition"

    2026.2 · Insper

Hi — I'm **Carlos Hernani**, an undergraduate student at Insper. This site is my
notebook for the *Artificial Neural Networks and Deep Learning* course: every
exercise I hand in, the code I actually ran to produce it, and the figures that
came out of that code.

Each report is self-contained. It states what was asked, shows the code, shows
the figures, and then says what the results mean — so it can be read without
opening a notebook, and re-run by anyone who clones the repo.

## What's here

<div class="grid cards" markdown>

- :material-chart-scatter-plot: **[Data](./exercises/data/index.md)**

    Class separability in 2D, non-linearity in 5D, and preprocessing the
    Spaceship Titanic dataset for `tanh` hidden units.

- :material-vector-line: **[Perceptron](./exercises/perceptron/index.md)**

    A single linear unit and the boundaries it can — and cannot — learn.

- :material-graph-outline: **[MLP](./exercises/mlp/index.md)**

    Hidden layers, backpropagation, and what depth buys over a single unit.

- :material-shuffle-variant: **[VAE](./exercises/vae/index.md)**

    Variational autoencoders: latent space, reconstruction and the KL term.

</div>

Longer work lives under **[Projects](./projects/index.md)**.

## Status

- [x] Data
- [ ] Perceptron
- [ ] MLP
- [ ] VAE
- [ ] Project

## Running the code

Everything in this repository runs from a single virtual environment:

``` shell
python -m venv env
source ./env/bin/activate      # Windows: .\env\Scripts\activate
python -m pip install -r requirements.txt --upgrade
```

To serve this site locally:

<!-- termynal -->

``` shell
mkdocs serve -o
```

## Source

The repository is at
[github.com/carloshernanic/ann-dl](https://github.com/carloshernanic/ann-dl){:target="_blank"}.
Reports live in `docs/`, the sources that produced each report live next to it
in a `code/` folder, and the figures in `figures/`.
