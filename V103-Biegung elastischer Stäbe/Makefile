

NAME=Protokoll

all:  Aufbau_Ablauf.tex Auswertung.tex Diskussion.tex Fehlerrechnung.tex Regression.tex Theorie.tex $(NAME).tex
	pdflatex $(NAME).tex
	pdflatex $(NAME).tex
	pdflatex $(NAME).tex
	rm -rf *.aux
	rm -rf *.log
	rm -rf *.idx
	rm -rf *.synctex.gz

clean:
	rm -rf *.aux
	rm -rf *.log
	rm -rf *.idx
	rm -rf *.toc
	rm -rf *.synctex.gz

clean-all:
	rm -rf *.aux
	rm -rf *.log
	rm -rf *.idx
	rm -rf *.synctex.gz
	rm -rf $(NAME).pdf
	rm -rf *.toc
