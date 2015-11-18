NAME=Protokoll

all:  Aufbau_Ablauf.tex Auswertung.tex Diskussion.tex Fehlerrechnung.tex Regression.tex Theorie.tex $(NAME).tex Titlepage.tex
	pdflatex $(NAME).tex
	pdflatex $(NAME).tex
	biber $(NAME).bcf
	pdflatex $(NAME).tex
	rm -rf *.aux
	rm -rf *.log
	rm -rf *.idx
	rm -rf *.synctex.gz
	rm -rf *.toc
	rm -rf *.bbl
	rm -rf *.bcf
	rm -rf *.bfc.blg
	rm -rf *.blg
	rm -rf *.run.xml
	rm -rf *.tex.blg

clean:
	rm -rf *.aux
	rm -rf *.log
	rm -rf *.idx
	rm -rf *.synctex.gz
	rm -rf *.toc
	rm -rf *.bbl
	rm -rf *.bcf
	rm -rf *.bfc.blg
	rm -rf *.blg
	rm -rf *.run.xml
	rm -rf *.tex.blg


clean-all:
	rm -rf *.aux
	rm -rf *.log
	rm -rf *.idx
	rm -rf *.synctex.gz
	rm -rf *.toc
	rm -rf $(NAME).pdf
	rm -rf *.bbl
	rm -rf *.bcf
	rm -rf *.bfc.blg
	rm -rf *.blg
	rm -rf *.run.xml
	rm -rf *.tex.blg

