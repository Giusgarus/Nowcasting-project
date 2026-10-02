# Thesis Defense

English, 16:9 LaTeX Beamer presentation: 17 main slides (including the closing
screen), approximately 19 minutes, followed by ten optional Q&A backup slides.
Berlin provides clickable section labels and slide markers across five sections:
Introduction, Methodology, Models, Results, and Conclusions. The plain title
and closing slides are included in the 17-slide count; content-slide footers
retain the numeric counter and UniPi logo. Compile with `latexmk` so navigation
is updated across passes.

## Backup Slides


Edit `backup_slides.tex` for the optional material after the Thank-you screen.
The main file inserts it after `\appendix`, which starts a separate Beamer part,
so its pages do not add markers to the main Berlin navigation. Each backup uses
`noframenumbering`, keeping the main denominator at 17. Keep that option on any
new backup frame. The backup header navigation is hidden and its footer says
"Backup material", with a link back to the closing screen. The closing screen
also links to the first backup.

Topics: exact switch rules; Perfect and post-processing; models and inputs;
search scope; normalization and imputation; event definitions and censoring;
duration error versus switch quality; evaluation grids; survival-model outputs;
survival results and statistical uncertainty. No additional experiments are run.

## Printable Speaker Notes

`output/Giuseppe_Gabriele_Russo_Speaker_Notes.pdf` is the A4, black-and-white
speaking copy: English spoken text, slide headings, target timings and
transitions, followed by optional Q&A notes. It excludes layout instructions
and on-screen text already visible in the deck. The source is
`presentation_plan.md`; regenerate after changing the speech.

From this folder, using the project's Python environment:

```bash
python scripts/export_speaker_notes.py
latexmk -norc -pdf -outdir=build -jobname=Giuseppe_Gabriele_Russo_Speaker_Notes -interaction=nonstopmode -halt-on-error speaker_notes.tex
cp build/Giuseppe_Gabriele_Russo_Speaker_Notes.pdf output/
```

`-norc` prevents the presentation's local build configuration from reusing its
PDF name. The separate existing `presentation_print.tex` remains untouched;
it prints the full plan rather than the speech-only version.

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
logo, adjusting the `../figures/cherubino_pant541.png` references if needed.
