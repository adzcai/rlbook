# cs-stat-184-notes

For the Harvard undergraduate course **CS/STAT 184: Introduction to Reinforcement Learning**.

Run `git clean -iX` to remove extra files not tracked by Git repository.

Written using [Markdown](https://www.markdownguide.org/) and [Jupyter Book](#about-jupyter-book).

## Overleaf notes

Add the remote <https://git.overleaf.com/63e2a72f2cf8b91fc96f46a6> and push there to update the Overleaf.

```
git remote add overleaf https://git.overleaf.com/63e2a72f2cf8b91fc96f46a6
git push overleaf main:master
```

# Getting started

Create a new [mamba](https://mamba.readthedocs.io/en/latest/index.html) environment (or [conda](https://docs.conda.io/projects/conda/en/stable/) if you prefer):

```
mamba create -f environment.yml
mamba activate rlbook
jb build book
open book/_build/html/index.html
```

[_config.yml](_config.yml) contains project configuration.

[_toc.yml](_toc.yml) contains the table of contents.

## About Jupyter Book

[Jupyter Book](https://jupyterbook.org/en/stable/intro.html) is a framework for building books from Jupyter Notebooks. It is essentially a bundle of Sphinx extensions that enable parsing Markdown (of the MyST flavor described below) and running Juypter Notebook code. [Sphinx](https://www.sphinx-doc.org/en/master/index.html) is a popular engine for generating documentation. It is used by many popular software projects (e.g. the [Flask](https://flask.palletsprojects.com) documentation). See [how Jupyter Book and Sphinx relate to one another](https://jupyterbook.org/en/stable/explain/sphinx.html).

[MyST (Markedly Structured Text)](https://myst-parser.readthedocs.io/en/latest/index.html) is a superset of [CommonMark](https://commonmark.org/) that supports tables, figures, etc., and especially a powerful system of [directives](https://myst-parser.readthedocs.io/en/latest/syntax/roles-and-directives.html). These include the original [Sphinx directives](https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html) as well as the `code-cell` directive that allows MyST documents to be run as Jupyter notebooks via [JupyText](https://jupytext.readthedocs.io/en/latest/#). For version control purposes, this is much more convenient than storing native Jupyter notebooks. MyST Markdown is not yet supported within the traditional Jupyter interface.

## MyST syntax

- [Math and equations](https://jupyterbook.org/en/stable/content/math.html)
  - Enclose inline math in dollar signs. `$1 + 2 = 3$`
  - Use the `math` directive instead of LaTeX `\label`s
- [Proofs, Theorems, and Algorithms](https://jupyterbook.org/en/stable/content/proof.html)
  - [Sphinx Proof](https://sphinx-proof.readthedocs.io/en/latest/)
    - To cite an object, use the `prf:ref` inline role instead of Markdown reference
    - Supports `proof`, `theorem`, `axiom`, `lemma`, `definition`, `criterion`, `remark`, `conjecture`, `corollary`, `algorithm`, `example`, `property`, `observation`, `proposition`, `assumption`
  - [Defining TeX macros](https://jupyterbook.org/en/stable/advanced/sphinx.html#defining-tex-macros)
- [Direct LaTeX Math](https://myst-parser.readthedocs.io/en/latest/syntax/optional.html#direct-latex-math)
  - `equation, multline, gather, align, alignat, flalign, matrix, pmatrix, bmatrix, Bmatrix, vmatrix, Vmatrix, eqnarray` will be directly parsed (no dollar signs required) and will work inside markdown components but cannot be labelled (needs the `math` directive)
- [Store code outputs and insert into content](https://jupyterbook.org/en/stable/content/executable/output-insert.html)
- [Notebooks written entirely in Markdown](https://jupyterbook.org/en/stable/file-types/myst-notebooks.html)
  - Converting Markdown to Jupytext:

```bash
jb myst init mymarkdownfile.md
```

- [Publish your book online with GitHub Pages](https://jupyterbook.org/en/stable/start/publish.html#publish-your-book-online-with-github-pages)

```bash
ghp-import -n -p -f book/_build/html
```

## Markdown syntax

[Docs](https://jupyterbook.org/en/stable/content/references.html#content-references)

Citations are kept in the [references.bib](./references.bib) file.

I recommend the [MyST-Markdown](https://marketplace.visualstudio.com/items?itemName=ExecutableBookProject.myst-highlight) VS Code extension.

````md
(my-label)=
## My header

Some text that refers to [the header above](my-label) or also to [another file](../1_topic/foobar.md).

And cites multiple papers like {cite}`silver_mastering_2016,mnih_playing_2013` using their citekeys from the BibTeX file

Figures and tables need to have a `:name:` property:

```{figure} ../path/to/image.jpg
:name: fig-label

This is the caption, which must exist in order for this figure to be referenced
```

Images can't be referenced:

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

And then refer to these using {eq}`my-equation` (note this uses the `eq` role)

Proofs and theorems are under the `prf` directive:

```{prf:theorem} My theorem
:label: my-theorem

This is the content of the theorem
```

And then refer to these using {prf:ref}`my-theorem` (note this uses the `prf:ref` role)
````


## Code syntax

- We use [Plotly Express](https://plotly.com/python/plotly-express/) for plotting.


