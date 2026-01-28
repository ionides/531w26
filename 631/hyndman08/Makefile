default: pdf-docs

pdf-docs: *.pdf

%.pdf: %.qmd
	quarto render "$*.qmd" --to beamer

clean:
	$(RM) *.log *.blg *.ilg *.aux *.lof *.lot *.toc *.idx
	$(RM) *.ttt *.fff *.out *.nav *.snm 
	$(RM) *.o *.so
	$(RM) *.brf
	$(RM) Rplots.ps Rplots.pdf

fresh: clean
	$(RM) *.ps *.bbl *.ind *.dvi 
	$(RM) -r cache/* figure/*
