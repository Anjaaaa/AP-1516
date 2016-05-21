import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import scipy.constants as const
from uncertainties import ufloat
from table import(
	make_table,
	make_SI,
	write)

Strom1, Spannung1 = np.genfromtxt('Messung1.txt', unpack=True)
Strom2, Spannung2 = np.genfromtxt('Messung2.txt', unpack=True)
Anregungsspannung = np.genfromtxt('Messung3.txt', unpack=True)
Temperatur = np.genfromtxt('Messung_Temperatur.txt', unpack=True)



#Temperatuten umrechnen und Stöße in der Röhre bestimmen
Temperatur = Temperatur + const.zero_Celsius #celsius in kelvin
Sattigungsdruck = 5.5*10**7*np.exp(-6876/Temperatur) # p in mbar und T in kelvin
Weglange = 0.0029 / Sattigungsdruck # Weglange in cm und Druck in mabr
#Stöße pro ein cm (Röhrenlänge) ausrechnen
Stosse = 1 / Weglange


write('build/tabelle_temperatur.tex', make_table([Temperatur, Sattigungsdruck, Weglange*10**4, Stosse], [2,2,2,2]))



#in Ampere und Volt umrechnen
Spannung1 = Spannung1*(10/228)
Strom1 = Strom1*(3.8e-6/143)
Spannung2 = Spannung2*(9.6/217)
Strom2 = Strom2*(1.1e-7/86)
Anregungsspannung = Anregungsspannung*(27/222)


matrix1 = np.ones((2, len(Spannung1)-1))

# matrix_x[0:i] ist der strom pro spannung, matrix_x[1:i] der zugehörige stonnungswert

for i in range(0, len(Spannung1)-1):
	matrix1[0,i] = (Strom1[i+1] -Strom1[i])/(Spannung1[i]-Spannung1[i+1])
	matrix1[1,i] = Spannung1[i]
	
plt.plot(matrix1[1,:], matrix1[0,:]*10**9, 'ro')
plt.xlabel('Spannung / V')
plt.ylabel('Steigung des Stromes / nA/V')
plt.savefig('build/Energieverteilung_25.png')
plt.show()

write('build/tabelle_stromverlauf_25.tex', make_table([Spannung1, Strom1*10**9], [2,0]))
write('build/tabelle_energieverteilung_25.tex', make_table([matrix1[1,:], matrix1[0,:]*10**9], [2,2]))


matrix2 = np.ones((2, len(Spannung2)-1))

for i in range(0, len(Spannung2)-1):
	matrix2[0,i] = (Strom2[i+1]-Strom2[i])/(Spannung2[i]-Spannung2[i+1])
	matrix2[1,i] = Spannung2[i]
	
	
	
plt.plot(matrix2[1,:], matrix2[0,:]*10**9, 'ro')
plt.xlabel('Spannung / V')
plt.ylabel(r'Steigung des Stromes / nA/V')
plt.savefig('build/Energieverteilung_140.png')
plt.show()

write('build/tabelle_stromverlauf_140.tex', make_table([Spannung2, Strom2*10**9], [2,0]))
write('build/tabelle_energieverteilung_140.tex', make_table([matrix2[1,:], matrix2[0,:]*10**9], [2,2]))

# Berechnung der Austrittsenergie	

Anregungsspannung_gesamt = ufloat(np.mean(Anregungsspannung), sem(Anregungsspannung))

Energie = const.e * Anregungsspannung_gesamt

Wellenlange = const.h*const.c / Energie

write('build/tabelle_anregungsspannung.tex', make_table([Anregungsspannung], [4]))

write('build/Anregungsspannung.tex', make_SI(Anregungsspannung_gesamt, r'\volt', figures=2))

write('build/Energie.tex', make_SI(Energie*10**19, r'\joule','e-19',figures=2))

write('build/Wellenlange.tex', make_SI(Wellenlange*10**9, r'\nano\meter', figures=2))

print(Anregungsspannung_gesamt, Energie, Wellenlange)





	