

NAME=Protokoll

all:  Aufbau_Ablauf.tex Auswertung.tex Diskussion.tex Fehlerrechnung.tex Regression.tex Theorie.tex $(NAME).tex
	pdflatex $(NAME).tex
	pdflatex $(NAME).tex
	pdflatex $(NAME).tex
	rm -rf *.aux
	rm -rf *.log
	rm -rf *.idx
	rm -rf *.synctex.gz
	rm -rf *.toc

clean:
	rm -rf *.aux
	rm -rf *.log
	rm -rf *.idx
	rm -rf *.synctex.gz
	rm -rf *.toc

clean-all:
	rm -rf *.aux
	rm -rf *.log
	rm -rf *.idx
	rm -rf *.synctex.gz
	rm -rf *.toc
	rm -rf $(NAME).pdf
