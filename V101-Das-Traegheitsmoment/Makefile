

NAME=Protokoll

all:  Aufbau_Ablauf.tex  Auswertung.tex  $(NAME).tex  Theorie.tex Tabelle_1.tex Tabelle_2.tex Tabelle_3.tex Tabelle_4.tex Tabelle_5.tex
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
	rm -rf *.synctex.gz

clean-all: 
	rm -rf *.aux
	rm -rf *.log
	rm -rf *.idx
	rm -rf *.synctex.gz
	rm -rf $(NAME).pdf
