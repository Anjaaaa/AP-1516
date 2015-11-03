

NAME=Protokoll

all:  Aufbau_Ablauf.tex Auswertung.tex Diskussion.tex Fehlerrechnung.tex Geschwindigkeit.tex Mittelwert_Zeitmessung.tex Quelle_bewegt_sich_hin.tex Quelle_bewegt_sich_weg.tex Regression.tex Ruhefrequenz.tex Theorie.tex Wellenlange.tex Wertepaare_hin.tex Wertepaare_weg.tex Zeitmessung.tex $(NAME).tex
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
