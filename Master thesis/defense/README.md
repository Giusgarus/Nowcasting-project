# Thesis Defense

English, 16:9 LaTeX Beamer presentation: 16 main slides, approximately 19 minutes.
Berlin provides clickable section labels and slide markers across five sections:
Introduction, Methodology, Models, Results, and Conclusions. The plain title
slide is included in the 16-slide count; the footer retains the numeric counter
and UniPi logo. Compile with `latexmk` so navigation is updated across passes.

## Compile

From this folder:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error defense.tex
```

The local `.latexmkrc` keeps compilation files in `build/` and copies the final
PDF to `output/Giuseppe_Gabriele_Russo_Defense.pdf`. It does not compile or change
the thesis. Speaker notes and timing are in `presentation_plan.md`.

## Assets and Evidence

- UniPi logo: `../figures/cherubino_pant541.png`; MBI logo: `mbi.png`.
- The opening slide uses both logos; subsequent slides show only UniPi in the footer.
- Small chart tables are stored in `data/`, with source paths in `provenance.json`.
- To refresh chart data from the existing experimental artifacts, run
  `conda run -n Nowcasting python "Master thesis/defense/scripts/export_slide_data.py"`
  from the repository root. This reads results without rerunning experiments.
- Native metrics and their interpretation follow the approved presentation plan
  and thesis Chapter 6. Common-grid metrics must not be mixed with native scores.
- The survival switch threshold of 0.65 is explicitly exploratory/test-selected.

To move the presentation alone to Overleaf, include this folder and the UniPi
logo, adjusting the two `../figures/cherubino_pant541.png` references if needed.
