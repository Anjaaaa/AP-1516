import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties import ufloat
from tabulate import tabulate

t, U = np.genfromtxt('4a.txt', unpack=True)
t = t*10**-6 #Mikrosekunden in Sekunden
U_0 = 19.4
t_err = 3*10**-5

U_fehler = np.zeros(len(U))

for i in range(0,len(U)):
	U_fehler[i] = 0.02
	
U_gesamt = unp.uarray(U, U_fehler)
print(U_gesamt)
print(unp.std_devs(U_gesamt)) 


plt.errorbar(t, U_0-U, xerr=t_err, yerr = 0.02, fmt='r.', label='Datenpunkte mit Messunsicherheit')
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

#Fehlerbalken in y-Richtung ausrechnen:
U_err = unp.log(U_0 - U_gesamt)

x=np.linspace(0,0.0035)
#plt.plot(t, U_regression, 'rx')
plt.errorbar(t, U_regression, xerr=t_err, yerr = unp.std_devs(U_err), fmt='r.', label='Datenpunkte mit Messunsicherheit')
plt.plot(x, f(x, *parameters), 'b-', label='Lineare Ausgleichsgerade')
plt.ylabel('$\log(U_0-U)$')
plt.xlabel('Zeit / s')
#plt.legend(loc='best')
plt.savefig('Spannung2.png')
plt.show()


print('Steigung der Geraden', m)
print('Achsenabschnitt der Geraden', b)
print('Zeitkonstante', Zeitkonstante.n, Zeitkonstante.s)
print('U_0 aus y-Achsenabschnitt', Ausgangsspannung)

#np.set_printoptions(precision=2)
tabelle = np.array([t* 10**3 , np.log(U_0-U), unp.std_devs(U_err)*100])
tabelle = np.around(tabelle, decimals=2)
f = open('tabelle1.tex', 'w')
print(tabulate(tabelle.T, tablefmt="latex"))
f.write(tabulate(tabelle.T, tablefmt="latex"))
