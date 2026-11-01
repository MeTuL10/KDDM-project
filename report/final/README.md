# Final report (IEEE conference format, 8 pages incl. references and appendix)

| File | Purpose |
|---|---|
| `main.tex` | The paper. Body is 6 pages; references and appendix follow. |
| `main.pdf` | Clean build (what gets submitted). |
| `main_tagged.pdf` | Same paper with a small grey `[tag]` after every empirical claim, for verification. |
| `claims.py`, `check_claims.py` | Registry mapping each tag to the source file and exact strings; the checker confirms every string occurs in its file. |
| `claims_ledger.md` | Generated ledger of all checks (tag, source file, string, status). |
| `make_figures.py`, `figures/` | New figures, built from logged numbers (leaderboard, learning curve, error concentration). |

Rebuild (from this folder, TeX Live on PATH):

    python3 make_figures.py          # regenerate figures
    python3 check_claims.py          # verify every cited number against its source file
    pdflatex main.tex && pdflatex main.tex                                   # clean PDF
    pdflatex -jobname=main_tagged "\def\SHOWSRC{}\input{main}"  (run twice)  # tagged PDF

Tag prefixes (internal, shown only in `main_tagged.pdf`): `Aa` V2/root pipelines, `Be` R pipeline, `Me` AutoGluon pipeline, `Pu` history pipeline, `Wk` workload.
Placeholders to fill: team name, three surnames, three student IDs (search `[` in `main.tex`).
