import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp
from tabulate import tabulate
from uncertainties.umath import *



t_klein, t_groß = np.genfromtxt('Daten_Zeit.txt', unpack = True)
T, t_T = np.genfromtxt('Daten_Temperatur.txt', unpack = True)
d_groß, d_klein = np.genfromtxt('Daten_Kugel.txt', unpack = True)


t_klein = ufloat(np.mean(t_klein), np.std(t_klein)/np.sqrt(len(t_klein)))
t_groß = ufloat(np.mean(t_groß), np.std(t_groß)/np.sqrt(len(t_groß)))
d_groß = ufloat(np.mean(d_groß), np.std(d_groß)/np.sqrt(len(d_groß))) / 1000
d_klein = ufloat(np.mean(d_klein), np.std(d_klein)/np.sqrt(len(d_klein))) / 1000

m_klein = 4.44 / 1000
m_groß = 4.63 / 1000


S = 0.10      # Fallweite
K_klein = 0.07640 / 1000


dichte_klein = m_klein / ( 4/3*np.pi*(d_klein/2)**3 )
dichte_groß = m_groß / ( 4/3*np.pi*(d_groß/2)**3 )

print('Dichte der kleinen Kugel:', dichte_klein)
print('Dichte der großen Kugel:', dichte_groß)


visk_20 = K_klein * (dichte_klein - 1) * t_klein
K_groß = visk_20 / (dichte_groß - 1) / t_groß

print('Viskosität bei 20 Grad:', visk_20)
print('Apparatekonstante große Kugel:', K_groß)

visk = (K_groß * (dichte_groß - 1)) * t_T


def  f(X, a, b):
    return a * X + b

X = 1/T
#Y = log(visk.n)
for i in range(0,len(visk)):
	print(visk[i].n)

#parameters, popt = curve_fit(f, X, Y)
#X = np.linspace(0, 0.07)
#plt.plot(X, f(X, parameters[0], parameters[1]), 'b-')
#plt.plot(X, Y, 'rx')


#plt.ylabel('$\ln(\eta)$')
#plt.xlabel('1/T')

#plt.savefig('')
#plt.show()



v = S / t_T

R = v * d_groß / visk
print(v)
print(R)
