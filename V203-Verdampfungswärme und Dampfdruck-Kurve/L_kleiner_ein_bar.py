import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from tabulate import tabulate

T, P = np.genfromtxt('Messreihe1.txt', unpack = True)

T = 273.15 + T
P = 100*P #mbar in Pascal

T = 1/T
P = np.log(P)


fit = np.polyfit(T, P, 1)
fit_fn = np.poly1d(fit)

def f(T, a, b):
    return a*T +b

parameters, popt = curve_fit(f, T, P)

a_gesamt = ufloat(parameters[0], np.sqrt(popt[0,0]))
R = 8.314 #gaskonstante
L = - a_gesamt * R
print (L)


#tabelle mit aufgetragenen daten
tabelle = np.array([T, P])
headers = ["Reziproke absolute Temperatur", "Logarithmus des Druckes"]

f = open('testtabelle.txt', 'w')
f.write(tabulate(tabelle.T, headers, tablefmt="latex"))

#f = open('tabelle.txt', 'w')
#f.write(tabulate(table, tablefmt="latex"))


plt.plot(T, P, 'rx', label = 'Datenpunkte')
plt.plot(T, fit_fn(T), 'g-', label='Regressionsfunktion')
plt.xlabel(r'$1/T \ /\  (\mathrm{1/K})$')
plt.ylabel(r'$log(P)$')
plt.legend(loc='best')
plt.savefig('L_kleiner_Druck.pdf')
plt.show()
