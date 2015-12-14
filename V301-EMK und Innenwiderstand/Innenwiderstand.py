import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp
from tabulate import tabulate

I_b, U_b = np.genfromtxt('Messungb.txt', unpack=True)
I_c, U_c = np.genfromtxt('Messungc.txt', unpack=True)
I_d, U_d = np.genfromtxt('Messungd_rechteck.txt', unpack=True)
I_e, U_e = np.genfromtxt('Messungd_sinus.txt', unpack=True)

I_d = I_d / 1000
I_e = I_e / 1000
U_e = U_e / 1000



def f(I_b, R, Uo):
    return Uo - I_b * R

parameters, popt = curve_fit(f, I_b, U_b)
fit = np.polyfit(I_b, U_b, 1)
regression_b = np.poly1d(fit)

R_i = parameters[0] # für Aufgabe 4c,d)
Uo = parameters[1] #für Aufgabe 4d)

print('Leerlaufspannung Messung B:', parameters[1], '+/-', np.sqrt(popt[1,1]))
print('Innenwiderstand Messung B:', parameters[0], '+/-', np.sqrt(popt[0,0]))
x = np.linspace(0.00,0.3)
plt.plot(x, regression_b(x), 'b-', label='Regressionsgerade')
plt.errorbar(I_b, U_b, xerr=I_b*0.03, yerr=U_b*0.02, fmt='r.', label='Datenpunkte')
plt.xlabel('Stromstärke $\mathrm{I} \ /\  \mathrm{A}$')
plt.ylabel('Spannung $\mathrm{U} \ /\  \mathrm{V}$')
plt.legend(loc='best')
plt.savefig('Spannung_Messung_b.pdf')
plt.show()

def f(I_c, R, Uo):
    return Uo + I_c * R

parameters, popt = curve_fit(f, I_c, U_c)
fit = np.polyfit(I_c, U_c, 1)
regression_c = np.poly1d(fit)

print('Leerlaufspannung Messung c:', parameters[1], '+/-', np.sqrt(popt[1,1]))
print('Innenwiderstand Messung c:', parameters[0], '+/-', np.sqrt(popt[0,0]))

y = np.linspace(0.01,0.4)
plt.plot(y, regression_c(y), 'b-', label = 'Regressionsgerade')
plt.xlim(0.01,0.4)
plt.errorbar(I_c, U_c, xerr=I_c*0.03, yerr=U_c*0.02, fmt='r.', label = 'Datenpunkte')
plt.xlabel('Stromstärke $\mathrm{I} \ /\  \mathrm{A}$')
plt.ylabel('Spannung $\mathrm{U} \ /\  \mathrm{V}$')
plt.legend(loc='best')
plt.savefig('Spannung_Messung_c.pdf')
plt.show()

def f(I_d, R, Uo):
    return Uo - I_d * R

parameters, popt = curve_fit(f, I_d, U_d)
fit = np.polyfit(I_d, U_d, 1)
regression_d = np.poly1d(fit)

print('Leerlaufspannung Messung d (Rechteck):', parameters[1], '+/-', np.sqrt(popt[1,1]))
print('Innenwiderstand Messung d', parameters[0], '+/-', np.sqrt(popt[0,0])) #Achtung I in mA gemessen

z = np.linspace(0.001,0.007)
plt.plot(z, regression_d(z), 'b-', label='Regressionsgerade')
plt.errorbar(I_d, U_d, xerr=I_d*0.03, yerr=U_d*0.02, fmt='r.', label='Datenpunkte')
plt.xlabel('Stromstärke $\mathrm{I} \ /\  \mathrm{A}$')
plt.ylabel('Spannung $\mathrm{U} \ /\  \mathrm{V}$')
plt.legend(loc='best')
plt.xlim(0.001,0.007)
plt.savefig('Spannung_Messung_d.pdf')
plt.show()

def f(I_e, R, Uo):
    return Uo - I_e * R

parameters, popt = curve_fit(f, I_e, U_e)
fit = np.polyfit(I_e, U_e, 1)
regression_e = np.poly1d(fit)

print('Leerlaufspannung Messung e (Sinus):', parameters[1], '+/-', np.sqrt(popt[1,1]))
print('Innenwiderstand Messung e:', parameters[0], '+/-', np.sqrt(popt[0,0]))


q = np.linspace(0,0.00035)
plt.plot(q, regression_e(q), 'b-', label ='Regressionsgerade')
plt.errorbar(I_e, U_e, xerr=I_e*0.03, yerr=U_e*0.02, fmt='r.', label = 'Datenpunkte')
plt.xlabel('Stromstärke $\mathrm{I} \ /\  \mathrm{A}$')
plt.ylabel('Spannung $\mathrm{U} \ /\  \mathrm{V}$')
plt.xlim(0,0.00035)
plt.legend(loc='best')
plt.savefig('Spannung_Messung_e.pdf')
plt.show()


#Aufgabe 4c)
deltaUo = 1.5 * (R_i / 10000000)
print('Aufgabe 4c: Zu Messung a ist die systematische Abwichung:', deltaUo)
print('was', deltaUo/1.5, '% entspricht.')

#Aufgabe 4e)

I_b = unp.uarray(I_b, I_b*0.03)
U_b = unp.uarray(U_b, U_b*0.02)

Leistung = I_b * U_b
Belastungswiderstand = U_b / I_b

tabelle = np.array([unp.nominal_values(Belastungswiderstand), unp.std_devs(Belastungswiderstand), unp.nominal_values(Leistung), unp.std_devs(Leistung)])

f = open('tabelle.txt', 'w')
f.write(tabulate(tabelle.T, tablefmt="latex"))



x = np.linspace(0,60, 100)
N = (Uo / (x + R_i))**2 * x # N=Leistung

for i in range (0, 10):
    print(unp.nominal_values(Leistung[i]), unp.std_devs(Leistung))

plt.errorbar(unp.nominal_values(Belastungswiderstand), unp.nominal_values(Leistung), xerr = unp.std_devs(Belastungswiderstand), yerr = unp.std_devs(Leistung),fmt = 'r.', label='Leistung aus Messdaten')
plt.plot(x, N, 'g-', label = 'Errechnete Leistungskurve')
plt.xlabel('Lastwiderstand $\mathrm{R_i} \ /\  \Omega$')
plt.ylabel('Leistung $\mathrm{N} \ /\  \mathrm{W}$')
plt.legend(loc='best')
plt.xlim(-1,60)
plt.savefig('Leistungskurve.pdf')
plt.show()
