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

D_links = D_ohne_links - D_mit_links  # Differenz berechnen
D_rechts = D_ohne_rechts - D_mit_rechts
D_links = D_links/1000   # von mm in m umrechnen
D_rechts = D_rechts/1000


I = 1   # Flächenträgheitsmoment
L = 0.50   # Länge des Stabes beim Einspannen
gewicht = 0.7476   # Gewicht des Gewichts nicht des Stabes

F = gewicht * 9.81


# x-Werte definieren

x_links = 0.50 - (0.2775 + x)   # Abstand zur linken Einspannung
x_rechts = 0.2775 - x

X_links = F/(48*I)*(3*L**2*x_links-4*x_links**3)
X_rechts = F/(48*I)*(3*L**2*x_rechts-4*x_rechts**3)



# Regression

print('LINKS')
fit = np.polyfit(X_links, D_links, 1)
fit_fn = np.poly1d(fit)

def f(X_links, m, b):
    return m*X_links+b

parameters, popt = curve_fit(f, X_links, D_links)

print('Steigung:', parameters[0])
print('Fehler Steigung:', np.sqrt(popt[1,1]))
print('y-Achsenabschnitt:', parameters[1])
print('Fehler y-Achsenabschnitt:', np.sqrt(popt[0,0]))
plt.plot(X_links, D_links, 'r.')
plt.plot(X_links, fit_fn(X_links), 'g-')
plt.show()



# Regression
print('RECHTS')
fit = np.polyfit(X_rechts, D_rechts, 1)
fit_fn = np.poly1d(fit)

def f(X_rechts, m, b):
    return m*X_rechts+b

parameters, popt = curve_fit(f, X_rechts, D_rechts)

print('Steigung:', parameters[0])
print('Fehler Steigung:', np.sqrt(popt[1,1]))
print('y-Achsenabschnitt:', parameters[1])
print('Fehler y-Achsenabschnitt:', np.sqrt(popt[0,0]))
plt.plot(X_rechts, D_rechts, 'r.')
plt.plot(X_rechts, fit_fn(X_rechts), 'g-')
plt.show()



# Regression
print('MITTELWERT')
X = (X_rechts + X_links)/2
D = (D_rechts + D_links)/2
fit = np.polyfit(X, D, 1)
fit_fn = np.poly1d(fit)

def f(X, m, b):
    return m*X+b

parameters, popt = curve_fit(f, X, D)

print('Steigung:', parameters[0])
print('Fehler Steigung:', np.sqrt(popt[1,1]))
print('y-Achsenabschnitt:', parameters[1])
print('Fehler y-Achsenabschnitt:', np.sqrt(popt[0,0]))
plt.plot(X, D_rechts, 'r.')
plt.plot(X, fit_fn(X), 'g-')
plt.show()

