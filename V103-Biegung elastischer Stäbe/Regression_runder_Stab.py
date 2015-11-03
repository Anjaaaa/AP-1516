import numpy as np
from uncertainties import ufloat
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp


# Abstand = x
# Durchbiegung mit Gewicht = D_mit
# Durchbiegung ohne Gewicht = D_ohne

x, D_mit, D_ohne = np.genfromtxt('Biegung_runder_Stab.txt', unpack = True)


x = x/100   # von cm in m umrechnen

D = D_ohne - D_mit  # Differenz berechnen
D = D/1000   # von mm in m umrechnen


I = 8.33*10**(-10)   # Flächenträgheitsmoment
L = 0.50   # Länge des Stabes beim Einspannen
gewicht = 0.7476   # Gewicht des Gewichts nicht des Stabes

F = gewicht * 9.81


# x-Werte definieren

X = F/(2*I)*(L*x**2-x**3/3)



# Regression

fit = np.polyfit(X, D, 1)
fit_fn = np.poly1d(fit)

def f(X, m, b):
    return m*X+b

parameters, popt = curve_fit(f, X, D)

print('Steigung:', parameters[0])
print('Fehler Steigung:', np.sqrt(popt[1,1]))
print('y-Achsenabschnitt:', parameters[1])
print('Fehler y-Achsenabschnitt:', np.sqrt(popt[0,0]))
print('Elastizitätsmodul:', 1/parameters[0])
plt.plot(X, D, 'r.')
plt.plot(X, fit_fn(X), 'g-')
plt.show()
