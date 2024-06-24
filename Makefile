ENV_NAME = rlbook

RUN = micromamba run -n $(ENV_NAME)

RAW_CHAPTERS = \
	1_intro/intro.md \
	2_bandits/bandits.md \
	3_mdps/mdps.md \
	4_fitted_dp/fitted_dp.md \
	5_control/control.md \
	6_pg/pg.md \
	7_exploration/exploration.md \
	appendix.md \
	bibliography.md \
	challenges.md \
	index.md

CHAPTERS = $(addprefix book/, $(RAW_CHAPTERS))

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
