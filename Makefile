book/_build/html: book/_config.yml
	jb build book

open: book/_build/html
	open book/_build/html/index.html

book/_build/latex: book/_config.yml
	jb build book --builder latex

pdf: book/_build/latex

clean: book/_build
	rm -r book/_build

debug:
	jb config sphinx book
