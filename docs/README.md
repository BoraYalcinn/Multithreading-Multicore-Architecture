# Reports

The written report for each lab is kept here in Markdown. Where a report was
submitted as a PDF, the PDF is stored alongside its Markdown source.

## Index

**Lab 1** — Thread fundamentals, synchronisation and process-based parallelism.
[`lab-1/`](lab-1/). In progress.

## Layout

Each lab has a directory containing `README.md`, which is the report itself, and
an `assets/` subdirectory for figures, screenshots and measurement data.
`REPORT_TEMPLATE.md` is the starting point for a new report:

    mkdir -p docs/lab-2/assets
    cp docs/REPORT_TEMPLATE.md docs/lab-2/README.md

Add a corresponding entry to the index above.

## Conventions

Timing results are reported as tables, preceded by a statement of the machine on
which they were taken: CPU model, physical and logical core count, compiler
version and optimisation flags. Every reported figure is the mean of at least
two or three runs; single measurements are not reported.

Figures are stored under the report's `assets/` directory and referenced by
relative path. References to code use repository-relative paths, for example
`../../Lab-1/Example4/RaceCondition.cpp`, so that they remain valid when the
report is read on GitHub.
