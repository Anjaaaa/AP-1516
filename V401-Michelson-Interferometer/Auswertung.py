import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp
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

Wellenlange_gesamt = ufloat(Wellenlange_Mittel, Wellenlange_Fehler)

print(Wellenlange_gesamt)


write('build/Wellenlange_gesamt.tex', make_SI(Wellenlange_gesamt,r'\meter','e-12', figures=4))


# Brechungsindex berechnen

b = 5e-3 #Kammernbreite in meter
T = 293.15 # Raumtemperatur in Kelvin
T_0 = 273.15 # Normaltemperatur in Kelvin
p_0 = 1.10132 # Normaldruck in bar

deltan_Luft = Wellenlange_gesamt *10**-12 * Impuls_Luft / 2 / b
n_Luft = 1 + deltan_Luft * (T/T_0)*(p_0/np.abs(Druck_Luft))

write('build/Tabelle4.tex', make_table([np.abs(Druck_Luft), Impuls_Luft, unp.nominal_values(n_Luft), unp.std_devs(n_Luft)],[1,0,8,8]))

deltan_CO2 = Wellenlange_gesamt *10**-12 * Impuls_CO2 / 2 / b
n_CO2 = 1 + deltan_CO2 * (T/T_0)*(p_0/np.abs(Druck_CO2))

print(n_Luft)
print(unp.nominal_values(n_Luft))
print(n_CO2)


