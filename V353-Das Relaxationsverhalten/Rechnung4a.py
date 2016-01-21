import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties import ufloat
from tabulate import tabulate

t, U = np.genfromtxt('4a.txt', unpack=True)
t = t*10**-6 #Mikrosekunden in Sekunden
U_0 = 20.0

plt.plot(t,U_0-U, 'rx', label='Spannung ($U_0-U$) \ Volt')
plt.ylabel('Spannung ($U_0-U$) / V')
plt.xlabel('Zeit / s')
plt.yscale('log')
plt.ylim(0.5, 30)
plt.savefig('Spannung1.png')
plt.show()


U_regression = np.log(U_0-U)
print(U_regression)
def f(t, m, b):
    return m*t+b

parameters, popt = curve_fit(f, t, U_regression)
m = ufloat(parameters[0], np.sqrt(popt[0,0]))
b = ufloat(parameters[1], np.sqrt(popt[1,1]))

Zeitkonstante = (-1/m)
Ausgangsspannung = unp.exp(b)

x=np.linspace(0,0.0035)
plt.plot(t, U_regression, 'rx')
plt.plot(x, f(x, *parameters), 'b-')
plt.ylabel('$\log(U_0-U)$')
plt.xlabel('Zeit / s')
plt.savefig('Spannung2.png')
plt.show()


print('Steigung der Geraden', m)
print('Achsenabschnitt der Geraden', b)
print('Zeitkonstante', Zeitkonstante.n, Zeitkonstante.s)
print('U_0 aus y-Achsenabschnitt', Ausgangsspannung)

#np.set_printoptions(precision=2)
tabelle = np.array([t* 10**3 , np.log(U_0-U)])
tabelle = np.around(tabelle, decimals=2)
f = open('tabelle1.tex', 'w')
f.write(tabulate(tabelle.T, tablefmt="latex"))
