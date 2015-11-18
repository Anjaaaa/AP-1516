import numpy as np
from uncertainties import ufloat
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp


# Abstand von der Mitte = x
# Durchbiegung mit Gewicht = D_mit
# Durchbiegung ohne Gewicht = D_ohne

x, D_mit_links, D_mit_rechts, D_ohne_links, D_ohne_rechts = np.genfromtxt('Biegung_beidseitig_eingespannt.txt', unpack = True)


x = x/100   # von cm in m umrechnen

D_links = D_ohne_links - D_mit_links          # Differenz berechnen
D_rechts = D_ohne_rechts - D_mit_rechts
D_links = D_links/1000                        # von mm in m umrechnen
D_rechts = D_rechts/1000


I = 8.33*10**(-10)   # Flächenträgheitsmoment
L = 0.555   # Länge des Stabes beim Einspannen
gewicht = 4.7004   # Gewicht des Gewichts, nicht des Stabes


F = gewicht * 9.81


# x-Werte definieren

x = 0.2775 - x   # Abstand zur Einspannung


X = (F/(48*I))*(3*L**2*x-4*x**3)         # Berechnung der x-Werte, die dann gegen die D-Werte aufgetragen werden



# Regression

print('LINKS')
fit_l = np.polyfit(X, D_links, 1)
fit_fn_l = np.poly1d(fit_l)

def f(X, m, b):
    return m*X+b

parameters_l, popt_l = curve_fit(f, X, D_links)

print('Steigung:', parameters_l[0])
print('Fehler Steigung:', np.sqrt(popt_l[0,0]))
print('y-Achsenabschnitt:', parameters_l[1])
print('Fehler y-Achsenabschnitt:', np.sqrt(popt_l[1,1]))
steigung_l = ufloat(parameters_l[0], np.sqrt(popt_l[0,0]))
print('Elastizitätsmodul:', 1/steigung_l)
plt.plot(X, D_links, 'r.', label='Datenpunkte')
plt.plot(X, fit_fn_l(X), 'g-', label = 'Regressionsgerade')

plt.xlabel(r'$C \ /\  \mathrm{N/m}$')
plt.ylabel(r'$D \ /\  \mathrm{m}$')

plt.legend(loc='best')
plt.savefig('Regression_zweiseitig_eingespannt_1')

plt.show()



# Regression

print('RECHTS')
fit_r = np.polyfit(X, D_rechts, 1)
fit_fn_r = np.poly1d(fit_r)

def f(X, m, b):
    return m*X+b

parameters_r, popt_r = curve_fit(f, X, D_rechts)

print('Steigung:', parameters_r[0])
print('Fehler Steigung:', np.sqrt(popt_r[0,0]))
print('y-Achsenabschnitt:', parameters_r[1])
print('Fehler y-Achsenabschnitt:', np.sqrt(popt_r[1,1]))
steigung_r = ufloat(parameters_r[0], np.sqrt(popt_r[0,0]))
print('Elastizitätsmodul:', 1/steigung_r)
plt.plot(X, D_rechts, 'r.', label = 'Datenpunkte')
plt.plot(X, fit_fn_r(X), 'g-', label = 'Regressionsgerade')

plt.xlabel(r'$C \ /\  \mathrm{N/m}$')
plt.ylabel(r'$D \ /\  \mathrm{m}$')

plt.legend(loc='best')
plt.savefig('Regression_zweiseitig_eingespannt_2.pdf')

plt.show()




# Regression
print('MITTELWERT')
D = (D_rechts + D_links)/2
fit = np.polyfit(X, D, 1)
fit_fn = np.poly1d(fit)

def f(X, m, b):
    return m*X+b

parameters_m, popt_m = curve_fit(f, X, D)

print('Steigung:', parameters_m[0])
print('Fehler Steigung:', np.sqrt(popt_m[1,1]))
print('y-Achsenabschnitt:', parameters_m[1])
print('Fehler y-Achsenabschnitt:', np.sqrt(popt_m[0,0]))
print('Elastizitätsmodul:', 1/parameters_m[0])
plt.plot(X, D_rechts, 'r.', label = 'Datenpunkte')
plt.plot(X, fit_fn(X), 'g-', label = 'Regressionsgerade')

plt.xlabel(r'$C \ /\  \mathrm{N/m}$')
plt.ylabel(r'$D \ /\  \mathrm{m}$')

plt.legend(loc='best')
plt.savefig('Regression_zweiseitig_eingespannt_3.pdf')

plt.show()


print('Schallgeschwindigkeit:', np.sqrt(1/(parameters_m[0]*2785)))
