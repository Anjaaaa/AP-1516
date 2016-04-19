import numpy as np
from uncertainties import ufloat
from table import (
	make_table,
	make_SI,
	write)

#Messung zur Bestimmung der Wellenlänge	
Anfang, Ende, Impuls = np.genfromtxt('Messung1.txt', unpack=True) 
#Messung mit CO2
Druck_CO2, Impuls_CO2 = np.genfromtxt('Messung2.txt', unpack=True)
#Messung mit Luft
Druck_Luft, Impuls_Luft = np.genfromtxt('Messung3.txt', unpack=True)



#Auswertung der Wellenlänge
#Hebelunterstellung beachten
a = 5.046 #Hebelunterschied
Anfang_2 = Anfang / a
Ende_2 = Ende / a 

write('build/Tabelle2.tex', make_table([Anfang, Ende, Anfang_2, Ende_2, Impuls],[2,2,2,2,0]))

#Wellenlänge jeder einzelnen Messung berechnen
Wellenlange = np.abs(Anfang_2 - Ende_2) / Impuls * 2

# Wellenlänge von milimeter in nanometer umrechnen
Wellenlange = Wellenlange *10**6

write('build/Tabelle3.tex', make_table([Anfang, Ende, Anfang_2, Ende_2, Impuls, Wellenlange],[2,2,2,2,0,1]))

#Durchscnittliche Wellenlange:
Wellenlange_Mittel = np.mean(Wellenlange)
Wellenlange_Fehler = np.std(Wellenlange, ddof=1)

print(Wellenlange_Mittel)
print(Wellenlange_Fehler)

write('build/Wellenlange_Mittel.tex', make_SI(Wellenlange_Mittel,r'\meter','e-12', figures=2))
write('build/Wellenlange_Fehler.tex', make_SI(Wellenlange_Fehler,r'\meter','e-12', figures=2))





print(Wellenlange)