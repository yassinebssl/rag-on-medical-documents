# Slides

- `cp1/main.tex`: Checkpoint 1 (2026-11-22), 7 slides for 5 minutes.
- `final/main.tex`: oral defense (2026-12-13), 15 slides for 20 minutes.

Each slide names its speaker in a `\note{}`. Red `TODO:` marks a value that is still missing.

## Compile

Run pdflatex twice from the deck's own folder. The final deck reads `../../report/generated/`, so the relative path only works from there.

```
cd docs/defense/final && pdflatex main.tex && pdflatex main.tex
cd docs/defense/cp1   && pdflatex main.tex && pdflatex main.tex
```

You need TeX Live with the beamer `metropolis` theme, `FiraSans`, `FiraMono`, `tikz`, `adjustbox` and `booktabs`.

The results slides read `docs/report/generated/retrieval_table.tex` and `summary_table.tex`. If a file is missing, the slide shows a TODO instead. Once the evaluation finishes, regenerate the tables and compile the deck again.

To print the speaker notes, add `\setbeameroption{show notes}` before `\begin{document}`.

## Still to do for the flash drive

`04_DEFENSE/` must contain `presentation.pdf` (the final `main.pdf`, renamed) and also `presentation.pptx`. `final/presentation.pptx` is the PowerPoint version of the same 15 slides, with the same speaker notes. It is built by a pptxgenjs script kept outside the repo (the course accepts `.py` scripts only): `node build_defense.js` in `~/Documents/ING5/GenAi/defense-pptx-build/`, with `NODE_PATH` set to its `node_modules`. It reads `results/retrieval.csv` and `results/summary.csv`, so rebuild it after the final evaluation run. Its TODOs are the same as the Beamer deck's.
