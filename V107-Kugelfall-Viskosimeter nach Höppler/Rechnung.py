import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp
from tabulate import tabulate
from uncertainties.umath import *



zeitKlein, zeitGroß = np.genfromtxt('Daten_Zeit.txt', unpack = True)
temp, zeitTemp1, zeitTemp2 = np.genfromtxt('Daten_Temperatur.txt', unpack = True)
dickeGroß, dickeKlein = np.genfromtxt('Daten_Kugel.txt', unpack = True)


zeitKlein = ufloat(np.mean(zeitKlein), np.std(zeitKlein)/np.sqrt(len(zeitKlein)))
zeitGroß = ufloat(np.mean(zeitGroß), np.std(zeitGroß)/np.sqrt(len(zeitGroß)))

print('Zeit klein:', zeitKlein)
print('Zeit groß:', zeitGroß)


temp = temp + 273.15
zeitTempMittel = (zeitTemp1 + zeitTemp2) / 2
zeitTempStd = np.sqrt((zeitTempMittel-zeitTemp1)**2 + (zeitTempMittel-zeitTemp2)**2)
zeitTemp = unp.uarray(np.zeros(len(zeitTempMittel)), np.zeros(len(zeitTempMittel)))
for i in range(0, len(zeitTempMittel)):
     zeitTemp[i] = ufloat(zeitTempMittel[i], zeitTempStd[i]/np.sqrt(2))

print('ZEIT:', zeitTemp)


dickeGroß = ufloat(np.mean(dickeGroß), np.std(dickeGroß)/np.sqrt(len(dickeGroß))) / 1000
dickeKlein = ufloat(np.mean(dickeKlein), np.std(dickeKlein)/np.sqrt(len(dickeKlein))) / 1000

print('Durchmesser Groß:', dickeGroß)
print('Durchmesser Klein:', dickeKlein)
print('Volumen Groß:', 4/3*np.pi*(dickeGroß/2)**3)
print('Volumen Klein:', 4/3*np.pi*(dickeKlein/2)**3)

masseKlein = 4.44 / 1000
masseGroß = 4.63 / 1000


dichteWasser = 992.8
fallweite = 0.10
visk20 = 1.002 / 1000

dichteKlein = masseKlein / ( 4/3*np.pi*(dickeKlein/2)**3 )
dichteGroß = masseGroß / ( 4/3*np.pi*(dickeGroß/2)**3 )

print('Dichte der kleinen Kugel:', dichteKlein)
print('Dichte der großen Kugel:', dichteGroß)


K_klein = visk20 / (dichteKlein - dichteWasser) / zeitKlein
K_groß = visk20 / (dichteGroß - dichteWasser) / zeitGroß

print('Apparatekonstante kleine Kugel:', K_klein)
print('Apparatekonstante große Kugel:', K_groß)

viskTemp = K_groß * (dichteGroß - dichteWasser) * zeitTemp

print('Viskosität Temp:', viskTemp)

def  f(X, a, b):
    return a * X + b

X = 1/temp
Y = np.zeros(len(X))

print('X:', X)

for i in range(0,len(X)):
	Y[i] = log(viskTemp[i].n)

print('Y:', Y)

parameters, popt = curve_fit(f, X, Y)
x = np.linspace((np.min(X)-0.00001), (np.max(X)+0.00001))
plt.plot(x*1000, f(x, parameters[0], parameters[1]), 'b-', label = 'Regressionsgerade')
plt.plot(X*1000, Y, 'rx', label = 'Werte')
plt.legend(loc='best')

plt.ylabel('$\ln(\eta)$')
plt.xlabel('$\degree C/T \ 10^{-3}$')

plt.savefig('Regression.png')
plt.show()


print('Steigung:', parameters[0], '+/-', np.sqrt(popt[0,0]))
print('y-Achsenabschnitt:', parameters[1], '+/-', np.sqrt(popt[1,1]))
A = np.exp(parameters[1])
AStd = np.exp(np.sqrt(popt[1,1]))
print('A:', A, '+/-', AStd)

vFluid = fallweite / zeitTemp
#reinold = (fallweite * dickeGroß * dichteWasser) / (zeitTemp**2 * K_groß * (dichteGroß - dichteWasser))
reinold = vFluid * dickeGroß * dichteWasser / viskTemp
print('Reinolds-Zahl:', reinold)
