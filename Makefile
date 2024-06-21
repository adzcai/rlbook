book/_build:
	jb build book

open: book/_build
	open book/_build/html/index.html

clean: book/_build
	rm -r book/_build

