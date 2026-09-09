# ANN & Deep Learning — Insper 2026.2

Exercises and projects for the *Artificial Neural Networks and Deep Learning*
course, published as a mkdocs site:
**<https://carloshernanic.github.io/ann-dl>**

## Layout

```
docs/
  index.md                  # home
  exercises/
    data/
      index.md              # the report
      code/                 # the sources that were actually run
      figures/              # the figures the report shows
    perceptron/
      index.md
      code/
      figures/
    mlp/index.md
    vae/index.md
  projects/
    index.md
mkdocs.yml
requirements.txt
```

Every exercise follows the same pattern: `index.md` is the report, `code/` holds
the scripts that produced it, and `figures/` holds the images the report embeds.

## Setup

Create a virtual environment:

``` shell
python3 -m venv env
```

Activate it (**every time you run anything from this repository**):

``` shell
source ./env/bin/activate       # Linux / macOS
.\env\Scripts\activate          # Windows (PowerShell)
```

Install the dependencies — this covers both the site and the exercise code:

``` shell
python3 -m pip install -r requirements.txt --upgrade
```

## Datasets

Datasets are **not** versioned here. Each report says where to get its data; the
notebooks expect it under `data/` at the root of the repository, which is
gitignored.

For the [Data](docs/exercises/data/index.md) exercise, download
[Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) from
Kaggle and unzip it into:

```
data/spaceship-titanic/train.csv
```

## Running the code

``` shell
python docs/exercises/data/code/ex1_clouds.py
python docs/exercises/data/code/ex2_nonlinearity.py
python docs/exercises/data/code/ex3_spaceship.py
```

Each script writes its figures to the sibling `figures/` folder and prints every
number quoted in the report. The scripts locate the repository root themselves,
so they can be run from any working directory.

## Documentation

Serve the site locally:

``` shell
mkdocs serve -o
```

Publish to GitHub Pages:

``` shell
mkdocs gh-deploy
```

Pushes to `main` also deploy automatically through
[`.github/workflows/main.yaml`](.github/workflows/main.yaml).
