ENV_NAME = rlbook

RUN = micromamba run -n $(ENV_NAME)

_NOTEBOOKS = $(addprefix book/, bandits control exploration fitted_dp imitation_learning mdps pg planning supervised_learning)

_META = \
	background \
	index

NOTEBOOKS = $(addsuffix .md, $(_NOTEBOOKS))

IPYNBS = $(addsuffix .ipynb, $(_NOTEBOOKS))

META = $(addsuffix .md, $(addprefix book/, $(_META)))

SOLUTIONS = book/solutions/bandits.py

SOURCE = $(NOTEBOOKS) $(META) $(SOLUTIONS)

CONFIG = book/_config.yml book/_toc.yml

book/_build/html: $(SOURCE) $(CONFIG)
	$(RUN) jb build -W -n --keep-going book

open: book/_build/html
	open book/_build/html/index.html

book/_build/latex: $(SOURCE) $(CONFIG)
	$(RUN) jb build book --builder latex

pdf: book/_build/latex
	cd book/_build/latex && make
	code book/_build/latex/book.log

clean: book/_build $(IPYNBS)
	rm -r book/_build
	rm $(IPYNBS)

debug:
	$(RUN) jb config sphinx book

sync: $(NOTEBOOKS)
	$(RUN) jupytext --sync $(NOTEBOOKS)

lab:
	$(RUN) jupyter lab

lint:
	$(RUN) ruff check --fix $(IPYNBS)

web:
	(cd book && $(RUN) myst build --html --execute)

publish:
	$(RUN) ghp-import --cname "rlbook.adzc.ai" --no-jekyll --push --force book/_build/html
