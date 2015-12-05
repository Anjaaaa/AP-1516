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
print('Steigung:', a_gesamt)
R = 8.314 #gaskonstante
L = - a_gesamt * R
print ('Verdampfungswärme:', L)


#tabelle mit aufgetragenen daten
tabelle = np.array([T, P])
headers = ["Reziproke absolute Temperatur", "Logarithmus des Druckes"]

f = open('testtabelle.txt', 'w')
f.write(tabulate(tabelle.T, headers, tablefmt="latex"))

#f = open('tabelle.txt', 'w')
#f.write(tabulate(table, tablefmt="latex"))


plt.plot(T*1000, P, 'rx', label = 'Datenpunkte')
plt.plot(T*1000, fit_fn(T), 'g-', label='Regressionsfunktion')
plt.xlabel(r'$\mathrm{T^{-1}} \ /\  \mathrm{K^{-1}}10^{-3}$')
plt.ylabel(r'$\ln(P)\ /\  \ln(\mathrm{Pa})$')
plt.locator_params(nbins=10)	# Anzahl der Striche an der x-Achse

plt.legend(loc='best')
plt.savefig('L_kleiner_Druck.pdf')
plt.show()

#vergleich innerer und äußerer verdampfungswärme
L_a = R * 373
L_i = L-L_a
print ('Innere Verdampfungswärme:', L_i)
print ('Äußere Verdampfungswärme:', L_a)
print ('Gesamtverpampfungswärme:', L)

print('L_i in eV/mol:', L_i *6.242e18)
print('Normaldruck?:', np.exp(parameters[1]))
