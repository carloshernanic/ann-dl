# Projects

!!! warning "Work in progress"

    Nothing here yet. Each project gets its own folder under `docs/projects/`,
    with the same layout as the exercises — an `index.md` report, a `code/`
    folder with the sources that were actually run, and a `figures/` folder with
    the images the report shows.

## Layout

```
docs/projects/
  index.md            # this page — index of the projects
  <project-name>/
    index.md          # the report
    code/             # the sources that were actually run
    figures/          # the figures the report shows
```

## Report outline

Every project report follows the same structure as the exercise reports:

1. **Objective** — the problem and why it matters.
2. **Data** — where it came from, how much there is, how it was prepared.
3. **Method** — architecture, training setup, hyperparameters.
4. **Results** — metrics and figures, each reproducible from `code/`.
5. **Discussion** — what worked, what did not, what would come next.
