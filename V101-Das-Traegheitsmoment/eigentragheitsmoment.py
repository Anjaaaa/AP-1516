import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat

a, T = np.genfromtxt('eigentragheitsmoment_Abstand und Winkel.txt', unpack=True)

a = a/1000 #mm in meter

a = a**2
T = T**2



fit = np.polyfit(a, T,1)
fit_fn = np.poly1d(fit)

def f(a, b, m):
    return m * a + b

parameters, popt = curve_fit(f, a, T)
print('y-Achsenabschnitt:', parameters[0])
print('Steigung:', parameters[1])
print('popt:', popt)


a_d1= sum(a)/len(a)
a_d2= np.average(a)

#tragheitsmoment korper:
gewicht, h, d = np.genfromtxt('eigentragheitsmoment_Zylinder.txt', unpack = True)

r = d/2
gewicht = gewicht[np.invert(np.isnan(gewicht))] /1000
h = h[np.invert(np.isnan(h))] /1000
r = r[np.invert(np.isnan(r))] /1000

R = np.average(r)
H = np.average(h)
m = gewicht

I_korper = 2* (m* (R**2 / 4 + H**2 / 12))

print('trägkeitmomente der Zylinder:', I_korper)

D = 0.02857 #aus winkelrichtgrose.py

I_D = parameters[0] * D / (4 * np.pi**2) - I_korper

print ('Eigenträgheitsmoment:', I_D)

print (popt)

y_fehler = 0
for i in range (0,len(a)):
    y = (T[i] - parameters[0] - parameters[1]*a[i])**2
    y_fehler = y_fehler + y



print('y-Varianz:', y_fehler / (len(a) - 2))
print('Fehlersumme:', y_fehler)

print('Fehler der Achsenabschnittes:',np.sqrt( y_fehler / 8 * sum(a**2) / ((len(a) * sum(a**2)) - 
sum(a)**2)) )
print('Fehler des Steigung:',np.sqrt( y_fehler / 8 * len(a) / ((len(a) * sum(a**2)) - sum(a)**2)))




plt.plot (a, T, 'r.')
plt.plot (a, fit_fn(a), 'g-')
#plt.show()


# Fehler Trägheitsmoment Zylinder

hoehe = ufloat(np.mean(h), np.std(h)/np.sqrt(len(h)))
radius = ufloat(np.mean(r), np.std(r)/np.sqrt(len(r)))
tragheitsmoment_zylinder = 2*m*(radius**2/4+hoehe**2/12)


print('Fehler Trägheitsmoment Zylinder:', tragheitsmoment_zylinder**2)


b = ufloat(5.18, 0.41)
fehler_eigenträgheitsmoment = b*0.0285742392151/(2*np.pi)**2-tragheitsmoment_zylinder

print('Eigenträgheitsmoment:', fehler_eigenträgheitsmoment)
