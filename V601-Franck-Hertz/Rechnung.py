import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import scipy.constants as const
from uncertainties import ufloat

Strom1, Spannung1 = np.genfromtxt('Messung1.txt', unpack=True)
Strom2, Spannung2 = np.genfromtxt('Messung2.txt', unpack=True)
Anregungsspannung = np.genfromtxt('Messung3.txt', unpack=True)


#in Ampere und Volt umrechnen
Spannung1 = Spannung1*(10/228)
Strom1 = Strom1*(3.8e-6/143)
Spannung2 = Spannung2*(9.6/217)
Strom2 = Strom2*(1.1e-7/86)
Anregungsspannung = Anregungsspannung*(27/222)


matrix1 = np.ones((2, len(Spannung1)-1))

for i in range(0, len(Spannung1)-1):
	matrix1[0,i] = (Strom1[i+1] -Strom1[i])/(Spannung1[i]-Spannung1[i+1])
	matrix1[1,i] = Spannung1[i]
	
plt.plot(matrix1[1,:], matrix1[0,:], 'ro')
plt.show()

matrix2 = np.ones((2, len(Spannung2)-1))

for i in range(0, len(Spannung2)-1):
	matrix2[0,i] = (Strom2[i+1]-Strom2[i])/(Spannung2[i]-Spannung2[i+1])
	matrix2[1,i] = Spannung2[i]
	
plt.plot(matrix2[1,:], matrix2[0,:], 'ro')
plt.show()


# Berechnung der Austrittsenergie	

Anregungsspannung_gesamt = ufloat(np.mean(Anregungsspannung), sem(Anregungsspannung))

Energie = const.e * Anregungsspannung_gesamt

Wellenlange = const.h*const.c / Energie

print(const.h, const.e, const.c)





	