# Report Scaffold

This directory contains a modular LaTeX scaffold for a conference-style report on efficient video moment retrieval.

Current default template:
- ACM-style via `acmart` using `sigconf` + `review` + `anonymous`
- Easy to swap later by editing only [main.tex](/f:/MS%20CS/IS/efficient-vmr/report/main.tex)

Suggested writing flow:
1. Edit [metadata.tex](/f:/MS%20CS/IS/efficient-vmr/report/metadata.tex) for the title and author block.
2. Write section content in `sections/`.
3. Add citations in `bib/references.bib`.
4. Put figures in `figures/` and modular tables in `tables/`.

Build on Windows PowerShell:

```powershell
Set-Location report
.\build.ps1
```

Clean build files:

```powershell
Set-Location report
.\build.ps1 -Clean
```

Direct `latexmk` usage also works:

```powershell
latexmk main.tex
```

Directory layout:

```text
report/
  main.tex
  metadata.tex
  macros.tex
  build.ps1
  latexmkrc
  bib/
    references.bib
  figures/
  sections/
    00_abstract.tex
    01_introduction.tex
    02_related_work.tex
    03_problem_setup.tex
    04_approach.tex
    05_system_design.tex
    06_experimental_setup.tex
    07_results.tex
    08_discussion.tex
    09_conclusion.tex
    appendix.tex
  tables/
    latency_breakdown.tex
    main_results.tex
```

Notes:
- The structure is tuned for CV + systems work, especially low-latency video moment retrieval.
- The current section split separates modeling choices from systems and latency analysis so you can emphasize both sides cleanly.
