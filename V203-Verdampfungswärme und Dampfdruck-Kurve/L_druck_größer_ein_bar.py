import numpy as np
from uncertainties import ufloat
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tabulate import tabulate


T, P = np.genfromtxt('Messreihe2.txt', unpack=True)

T = 273.15 + T
P = (7.03 - P)*10**5

print (T, P)


fit = np.polyfit(T, P, 3)
fit_fn = np.poly1d(fit)

def f(T, a, b, c, d):
    return a*T**3 + b*T**2 + c*T + d

parameters, popt = curve_fit(f, T, P)


plt.plot(T, P/1000, 'rx', label = 'Datenpunkte')
x = np.linspace(365, 405)
plt.plot(x, fit_fn(x)/1000, 'g-', label='Regressionsfunktion')
plt.xlabel(r'$T \ /\  \mathrm{K}$')
plt.ylabel(r'$P \ /\  \mathrm{Pa}\ 10^3$')
plt.legend(loc='best')
plt.savefig('Regerssionspolynom_P(t).pdf')
plt.show()


print('a:', parameters[0], np.sqrt(popt[0,0]))
print('b:', parameters[1], np.sqrt(popt[1,1]))
print('c:', parameters[2], np.sqrt(popt[2,2]))
print('d:', parameters[3], np.sqrt(popt[3,3]))
print(popt)


#Volumen berechnen
R = 8.314 #Gaskonstante
a = 0.9 #Koeffizient siehe Protokoll
V_plus = (R * T + np.sqrt(R**2 * T**2 - 4 * a * P)) / (2 * P)
V_minus = (R * T - np.sqrt(R**2 * T**2 - 4 * a * P)) / (2 * P)


plt.plot(T, V_plus)
plt.xlim(min(T), max(T))
plt.xlabel(r'$T \ /\  (\mathrm{K})$')
plt.ylabel(r'$V_m \ /\  (\mathrm{m^3/mol})$')
plt.savefig('Volumen+.pdf')
plt.show()

plt.plot(T, V_minus*1000)
plt.xlim(min(T), max(T))
plt.xlabel(r'$T \ /\  (\mathrm{K})$')
plt.ylabel(r'$V_m \ /\  (\mathrm{m^3/mol})\ 10^{-3}$')
plt.legend(loc='best')
plt.savefig('Volumen-.pdf')
plt.show()




V = V_plus #diese Werte sind realistischer!!

T_1 = parameters[0] * 3 * T**2 + parameters[1] * 2 * T + parameters[2] # Ableitung zu T

L = V * T * T_1

t = np.linspace(365, 405)

V_t = (R * t + np.sqrt(R**2 * t**2 - 4 * a *fit_fn(t))) / (2 * fit_fn(t))
P_1_t = parameters[0] * 3 * t**2 + parameters[1] * 2 * t + parameters[2]
L_t = V_t * t * P_1_t

tabelle = np.array([T, L])
headers = ["Temperatur in $\si{\kelvin}$", "Verdampfungswärme in $\si{\joule\per\mol}$"]

f = open('testtabelle2.txt', 'w')
f.write(tabulate(tabelle.T, headers, tablefmt="latex"))


plt.plot(t, L_t, 'g-', label= 'Regressionsfunktion')
plt.plot (T, L, 'rx', label ='Datenpunkte')
plt.xlabel(r'$T \ /\  \mathrm{K}$')
plt.ylabel(r'$L \ /\  (\mathrm{J/mol})$')
plt.legend(loc = 'best')
plt.savefig('L_groser_druck_temperaturabhangig.pdf')
plt.show()
