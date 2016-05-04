import numpy as np
from uncertainties import ufloat
from scipy import stats
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
Wellenlange_Fehler = stats.sem(Wellenlange)


Wellenlange_gesamt = ufloat(Wellenlange_Mittel, Wellenlange_Fehler)

print(Wellenlange_gesamt)


write('build/Wellenlange_gesamt.tex', make_SI(Wellenlange_gesamt,r'\meter','e-9', figures=2))


# Brechungsindex berechnen

b = 5*10**-2 #Kammernbreite in meter
T = 293.15 # Raumtemperatur in Kelvin
T_0 = 273.15 # Normaltemperatur in Kelvin
p_0 = 1.01325 # Normaldruck in bar

#deltan_Luft = Wellenlange_gesamt *10**-9 * Impuls_Luft / 2 / b
#n_Luft = 1 + deltan_Luft * (T/T_0)*(p_0/np.abs(Druck_Luft))


#korrigierte Version
a = Impuls_Luft * T / T_0 * p_0 / np.abs(Druck_Luft) /2 /b
a_gesamt = ufloat(np.mean(a), stats.sem(a))

print(a_gesamt)

n_Luft = 1 + Wellenlange_gesamt*10**-9 * a_gesamt
print(n_Luft)




write('build/Tabelle4_korrigiert.tex', make_table([np.abs(Druck_Luft), Impuls_Luft, a],[1,0,3]))

b = Impuls_CO2 * T / T_0 * p_0 / np.abs(Druck_CO2) /2 /b
b_gesamt = ufloat(np.mean(b), stats.sem(b))

n_CO2 = 1 + Wellenlange_gesamt*10**-9 * b_gesamt
print(n_CO2)



#deltan_CO2 = Wellenlange_gesamt *10**-9 * Impuls_CO2 / 2 / b
#n_CO2 = 1 + deltan_CO2 * (T/T_0)*(p_0/np.abs(Druck_CO2))

#write('build/Tabelle5.tex', make_table([np.abs(Druck_Luft), Impuls_Luft, unp.nominal_values(n_Luft), unp.std_devs(n_Luft)],[1,0,6,6]))

write('build/Tabelle6_korrigiert.tex', make_table([np.abs(Druck_CO2), Impuls_CO2, b],[1,0,3]))

write('build/a_Luft_korrigiert.tex', make_SI(a_gesamt, r'' ,figures=1))

write('build/a_CO2_korrigiert.tex', make_SI(b_gesamt, r'' ,figures=1))

write('build/Brechungsindex_Luft_korrigiert.tex', make_SI(n_Luft, r'' ,figures=1))

write('build/Brechungsindex_CO2_korrigiert.tex', make_SI(n_CO2, r'' ,figures=2))



# Mittelwert der Brechungsidizes mit Gewichtung des Fehlers berechnen:

#Mittel_n_Luft = sum(unp.nominal_values(n_Luft)/unp.std_devs(n_Luft)**2) / sum(1/unp.std_devs(n_Luft)**2)

#Fehler_n_Luft = np.sqrt(1 / sum(1/unp.std_devs(n_Luft)**2))

#print(sum(unp.nominal_values(n_Luft)/len(n_Luft)), Mittel_n_Luft)
#print(np.sqrt(sum(unp.std_devs(n_Luft))/len(n_Luft)), Fehler_n_Luft)

#write('build/Brechungsindex_Luft.tex', make_SI(ufloat(Mittel_n_Luft, Fehler_n_Luft), r'' ,figures=1))

#Mittel_n_CO2 = sum(unp.nominal_values(n_CO2)/unp.std_devs(n_CO2)**2) / sum(1/unp.std_devs(n_CO2)**2)

#Fehler_n_CO2 = np.sqrt(1 / sum(1/unp.std_devs(n_CO2)**2))

#print(sum(unp.nominal_values(n_CO2)/len(n_CO2)), Mittel_n_CO2)
#print(np.sqrt(sum(unp.std_devs(n_CO2))/len(n_CO2)), Fehler_n_CO2)

#write('build/Brechungsindex_CO2.tex', make_SI(ufloat(Mittel_n_CO2, Fehler_n_CO2), r'' ,figures=1))

