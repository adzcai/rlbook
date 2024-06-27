ENV_NAME = rlbook

RUN = micromamba run -n $(ENV_NAME)

_NOTEBOOKS = \
	1_intro/intro \
	2_bandits/bandits \
	3_mdps/mdps \
	4_fitted_dp/fitted_dp \
	5_control/control \
	6_pg/pg \
	7_exploration/exploration

NOTEBOOKS = $(addsuffix .md, $(addprefix book/, $(_NOTEBOOKS)))

IPYNBS = $(addsuffix .ipynb, $(addprefix book/, $(_NOTEBOOKS)))

_META = \
	appendix \
	bibliography \
	challenges \
	index

META = $(addsuffix .md, $(addprefix book/, $(_META)))

CHAPTERS = $(NOTEBOOKS) $(META)

CONFIG = book/_config.yml book/_toc.yml

book/_build/html: $(CHAPTERS) $(CONFIG)
	$(RUN) jb build book

open: book/_build/html
	open book/_build/html/index.html

book/_build/latex: $(CHAPTERS) $(CONFIG)
	$(RUN) jb build book --builder latex

pdf: book/_build/latex
	cd book/_build/latex && make
	code book/_build/latex/book.log

clean: book/_build
	rm -r book/_build

debug:
	$(RUN) jb config sphinx book

sync: $(NOTEBOOKS)
	$(RUN) jupytext --sync $(NOTEBOOKS)

lab:
	$(RUN) jupyter lab

lint:
	$(RUN) ruff check --fix $(IPYNBS)
