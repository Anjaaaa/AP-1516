import numpy as np
from uncertainties import ufloat


L = np.genfromtxt('Wellenlänge.txt', unpack = True)
f_0 = np.genfromtxt('Ruhefrequenz.txt', unpack = True)


L /= 1000   # gemessen in mm


# Hier zuerst die Mittelwerte ausrechnen, da wir nur vier Wellenlängen, aber fünf Ruhefrequenzen haben.

L = ufloat(np.mean(L), np.std(L)/np.sqrt(len(L)))
print('Wellenlänge:', L)
print('Inverses Wellenlänge:', 1/L)

f_0 = ufloat(np.mean(f_0), np.std(f_0)/np.sqrt(len(f_0)))
print('Ruhefrequenz:', f_0)
c = f_0*L
print('Schallgeschwindigkeit:', c)
