all: paper.pdf

paper.pdf: paper.latex
	pdflatex $<
	bibtex paper
	pdflatex $<
	pdflatex $<
