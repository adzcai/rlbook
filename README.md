# cs-stat-184-notes

For the Harvard undergraduate course **CS/STAT 184: Introduction to Reinforcement Learning**.

Run `git clean -iX` to remove extra files not tracked by Git repository.

Written using [Markdown](https://www.markdownguide.org/) and [Jupyter Book](https://jupyterbook.org/en/stable/content/proof.html)

## Overleaf notes

Add the remote <https://git.overleaf.com/63e2a72f2cf8b91fc96f46a6> and push there to update the Overleaf.

```
git remote add overleaf https://git.overleaf.com/63e2a72f2cf8b91fc96f46a6
git push overleaf main:master
```

# Contributing

Create a new [mamba](https://mamba.readthedocs.io/en/latest/index.html) environment (or [conda](https://docs.conda.io/projects/conda/en/stable/) if you prefer):

```
mamba create -f environment.yml
```

[_config.yml](_config.yml) contains project configuration.

[_toc.yml](_toc.yml) contains the table of contents.

Some relevant parts of the Jupyter Book documentation and related software:

- [Conda environment file](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-file-manually)
- [Sphinx](https://www.sphinx-doc.org/en/master/index.html): The (highly extensible) documentation engine that powers Jupyter Book.
  - [How Jupyter Book and Sphinx relate to one another](https://jupyterbook.org/en/stable/explain/sphinx.html)
    - "Jupyter Book can be thought of as an opinionated distribution of Sphinx"
  - Originally uses the [reStructuredText (rst)](https://docutils.sourceforge.io/rst.html) markup language but now supports [MyST Markdown](https://myst-parser.readthedocs.io/en/latest/index.html) (used by Jupyter Book), a superset of [CommonMark](https://commonmark.org/) that supports tables, figures, etc., and especially a powerful system of [directives](https://myst-parser.readthedocs.io/en/latest/syntax/roles-and-directives.html)
    - [Sphinx Directives](https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html)
- [Math and equations](https://jupyterbook.org/en/stable/content/math.html)
- [Proofs, Theorems, and Algorithms](https://jupyterbook.org/en/stable/content/proof.html)
  - [Sphinx Proof](https://sphinx-proof.readthedocs.io/en/latest/)
    - Need to use `prf:ref` inline role instead of Markdown reference
    - Supports `proof`, `theorem`, `axiom`, `lemma`, `definition`, `criterion`, `remark`, `conjecture`, `corollary`, `algorithm`, `example`, `property`, `observation`, `proposition`, `assumption`
  - [Defining TeX macros](https://jupyterbook.org/en/stable/advanced/sphinx.html#defining-tex-macros)
- [Direct LaTeX Math](https://myst-parser.readthedocs.io/en/latest/syntax/optional.html#direct-latex-math)
  - `equation, multline, gather, align, alignat, flalign, matrix, pmatrix, bmatrix, Bmatrix, vmatrix, Vmatrix, eqnarray` will be directly parsed (no dollar signs required) and will work inside markdown components but cannot be labelled (needs the `math` directive)
- [Store code outputs and insert into content](https://jupyterbook.org/en/stable/content/executable/output-insert.html)

## Syntax

[Docs](https://jupyterbook.org/en/stable/content/references.html#content-references)

Citations are kept in the [references.bib](./references.bib) file.

````md
(my-label)=
## My header

Some text that refers to [the header above](my-label) or also to [another file](../1_topic/foobar.md).

And cites multiple papers like {cite}`silver_mastering_2016,mnih_playing_2013` using their citekeys from the BibTeX file

Blocks need to have a `:name:` property:

```{figure} ../path/to/image.jpg
:name: fig-label

This is the caption, which must exist in order for this figure to be referenced
```

```{image} ../images/fun-fish.png
:alt: fishy
:class: bg-primary mb-1
:width: 200px
:align: center
```

```{table} The caption under the table
:name: my-table-ref

| first name | last name |
| --- | --- |
| Alexander | Cai |
```

Then I can refer to figures and tables using [the same syntax as before](fig-label) (or a reference to [that table](my-table-ref)) or also as {numref}`fig-label` (shows "Fig. 1") or {numref}`my-table-ref` (shows "Table 1")

```{math}
:label: my-equation

w^{t+1} = w^{t} - \nabla_w L(w^t)
```

And then refer to these using {eq}`my-equation`
````



