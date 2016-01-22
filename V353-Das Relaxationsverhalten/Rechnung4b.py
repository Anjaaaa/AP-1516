import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties import ufloat
from tabulate import tabulate


f, A, a = np.genfromtxt('4bc.txt', unpack = True)
a = a * 10**(-6) #mikrosekunde in sekunde
A = A/2 #Peak to Peak Amplitude gemessen
A_err = 0.01
f_err = 1


def  g(f, b, c, d):
    return b / (np.sqrt(1+(2*np.pi*f)**2*c)) +d

parameter, popt = curve_fit(g, f, A)

b = ufloat(parameter[0], np.sqrt(popt[0,0]))
c = ufloat(parameter[1], np.sqrt(popt[1,1]))
d = ufloat(parameter[2], np.sqrt(popt[2,2]))


print(b)
print(c)
print(d)

print('Ausgangsspannung', b)
print('Zeitkonstante', unp.sqrt(c))
print('Plotfehler', d)

print(popt)


x = np.linspace(0,1400)
plt.errorbar(f, A, xerr=f_err, yerr=A_err, fmt='r.')
#plt.plot(f, A, 'rx')
plt.plot(x, g(x, *parameter), 'b-')
plt.ylabel('Amplitude / V')
plt.xlabel('Frequenz / Hz')
plt.savefig('Amplitude.png')
plt.show()

tabelle = np.array([f , A])
tabelle = np.around(tabelle, decimals=2)
f = open('tabelle2.tex', 'w')
f.write(tabulate(tabelle.T, tablefmt="latex"))
