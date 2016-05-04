import numpy as np
import uncertainties
from uncertainties import ufloat
import uncertainties.unumpy as unp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sympy import *
from table import (
        make_table,
        make_SI,
        write)


### Reihenfolge der Farben: violett1, violett2, blau, türkis, grün, gelb, orange, rot
### phi = Winkel des Prismas, eta = Brechwinkel



wavelength, phi_1, phi_2, eta_2, eta_1 = np.genfromtxt('Werte_Sonja_Saskia.txt', unpack = True)

### Umrechnungen, nan wegmachen
wavelength = wavelength * 10**(-9)
phi_1 = phi_1[np.invert(np.isnan(phi_1))]
phi_2 = phi_2[np.invert(np.isnan(phi_2))]
print('eta_1:', eta_1)
print('eta_2:', eta_2)


################################################################
### etas und phi ausrechnen
################################################################
phi = 0.5 * (phi_1 - phi_2)
phi_Mittel = ufloat(np.mean(phi), np.std(phi)/np.sqrt(len(phi)))
# eta = 180 - (360 + eta_1 - eta_2)
eta = 180 - (eta_1-eta_2) # mit den Werten von Sonja und Saskia
print('eta:', eta)
print('phi_Mittel:', phi_Mittel)

phi_Mittel = 60

write('build/Messwerte1.tex', make_table([phi_1, phi_2, phi],[1,1,1]))
write('build/Winkel_Prisma.tex', make_SI(phi_Mittel,r'',figures=1))
write('build/Messwerte2.tex', make_table([wavelength*10**9, eta_1, eta_2, eta],[2,1,1,1]))
################################################################



################################################################
### Brechzahl
################################################################
n = unp.sin( 0.5 * 2 * np.pi / 360 * (eta + phi_Mittel) ) / unp.sin( 0.5 * 2 * np.pi / 360 * phi_Mittel)
print('Brechzahl:', n)


n_nom = unp.nominal_values(n)
n_std = unp.std_devs(n)


write('build/Brechzahlen.tex', make_table([wavelength*10**9, n_nom, n_std],[2,3,3]))
################################################################



################################################################
### Plotten
################################################################
### Berechnung der Hilfsgeraden
Y = n_nom**2
m = ( Y[0]-Y[len(Y)-1] ) / ( np.min(wavelength)-np.max(wavelength) )
b = Y[0] - m*wavelength[0]
def f(X,m,b):
     return m*X + b


### Wellenlänge -> n^2
plt.plot(wavelength*10**9, Y, 'kx', label='Messwerte')
plt.plot(wavelength*10**9, f(wavelength,m,b), 'r-', label = 'Hilfslinie')
#plt.plot(wavelength*10**9, n_nom, 'gx')

plt.xlabel(r'$\mathrm{\lambda} \ /\  \mathrm{nm}$')
plt.ylabel(r'$n^2$')
plt.legend(loc='best')

plt.savefig('Tendenz.png')
plt.show()



### NUR ZUR ANSCHAUUNG: Wellenlänge -> n und Wellenlänge -> Brechwinkel
#plt.plot(wavelength*10**9, n_nom, 'gx', label='Brechungsindex')
plt.plot(wavelength*10**9, eta/30, 'bx', label='Brechwinkel/30')


plt.xlabel(r'$\mathrm{\lambda} \ /\  \mathrm{nm}$')
plt.ylabel(r'$n$')
plt.legend(loc='best')
plt.show()
################################################################



################################################################
### Entscheidung für eine der beiden Gleichungen 11 bzw. 11a durch Regression
### WICHTIG! Strich hier ist gestrichen im Protokoll und nichts hier ist ungestrichen im Protokoll
################################################################
### wavelength^2 -> n^2
print('Regression mit Lambda^2')
# Hier mache ich die Wellenlänge größer, weil Python sonst ein Speicherproblem bekommt, weil die Zahlen so klein sind.
X = (wavelength*10**9)**2
regressionStrich = np.polyfit(X, Y, 1)
regFktStrich = np.poly1d(regressionStrich)

valuesStrich, errorsStrich = curve_fit(f, X, Y)


steigungStrich = 10**(-18)*ufloat(valuesStrich[0],np.sqrt(errorsStrich[0,0]))
verschStrich = ufloat(valuesStrich[1], np.sqrt(errorsStrich[1,1]))
print('Steigung:', steigungStrich)
print('y-Achsenabschnitt:', verschStrich)

plt.plot(X, Y, 'r.', label='Datenpunkte')
plt.plot(X, regFktStrich(X), 'g-', label = 'Regressionsgerade')

plt.xlabel(r'$\mathrm{\lambda^2} \ /\  \mathrm{m}^2$')
plt.ylabel(r'$n^2$')

write('build/RegStrichSteig.tex', make_SI(steigungStrich*10**(25),r'\meter^{-2}','e-25',figures=1))
write('build/RegStrichVersch.tex', make_SI(verschStrich,r'',figures=1))
write('build/RegStrichSteigRel.tex', make_SI(unp.std_devs(steigungStrich)/steigungStrich.n,r'\meter^{-2}',figures=1))
write('build/RegStrichVerschRel.tex', make_SI(unp.std_devs(verschStrich)/verschStrich.n,r'',figures=3))

plt.show()


### 1/wavelength^2 -> n^2
print('Regression mit 1/Lambda^2')
X = 1 / wavelength**2
regression = np.polyfit(X, Y, 1)
regFkt = np.poly1d(regression)

values, errors = curve_fit(f, X, Y)

steigung = ufloat(values[0],np.sqrt(errors[0,0]))
versch = ufloat(values[1], np.sqrt(errors[1,1]))
print('Steigung:', steigung)  
print('y-Achsenabschnitt:', versch)

plt.plot(X, Y, 'r.', label='Datenpunkte')
plt.plot(X, regFkt(X), 'g-', label = 'Regressionsgerade')

plt.xlabel(r'$\mathrm{\frac{1}{\lambda^2}} \ /\  \mathrm{m}^{-2}$')
plt.ylabel(r'$n^2$')

write('build/RegSteig.tex', make_SI(steigung*10**(14),r'\meter\squared','e-14',figures=1))
write('build/RegVersch.tex', make_SI(versch,r'',figures=1))
write('build/RegSteigRel.tex', make_SI(-unp.std_devs(steigung)/steigung.n,r'\meter\squared',figures=1))
write('build/RegVerschRel.tex', make_SI(unp.std_devs(versch)/versch.n,r'',figures=3))

plt.show()



### Abweichungsquadrate
abw = 1/6 * sum((n_nom**2 - versch.n - steigung.n / wavelength**2)**2)
abwStrich = 1/6 * sum((n_nom**2 - verschStrich.n - steigungStrich.n * wavelength**2)**2)
print('n_nom**2:', n_nom**2)
print('P_0:', versch.n)
print('P_2:', steigung.n/wavelength**2)
print('s**2:', abw)
print('s**2Strich:', abwStrich)

write('build/abwStrich.tex', make_SI(abwStrich*1000,r'','e-3',figures=1))
write('build/abw.tex', make_SI(abw*1000,r'','e-3',figures=1))



################################################################
### Endgültige Dispersionskurve
################################################################
def nFkt(wavelength):
     return np.sqrt( values[1]+values[0]/wavelength**2 )

plt.plot(wavelength*10**9, n_nom, 'k.', label='Datenpunkte')
plt.plot(wavelength*10**9, nFkt(wavelength), 'r-', label = 'Dispersionsfunktion')
plt.xlabel(r'$\mathrm{\lambda} \ /\  \mathrm{nm}$')
plt.ylabel(r'$n$')
plt.legend(loc='best')

plt.savefig('Dispersionskurve.png')
plt.show()



################################################################
### Abbesche Zahl
################################################################
C = 656 * 10**(-9)
D = 589 * 10**(-9)
F = 486 * 10**(-9)

print('n(C):', nFkt(C))
print('n(D):', nFkt(D))
print('n(F):', nFkt(F))



abbe = ( nFkt(D)-1 ) / ( nFkt(F)-nFkt(C) )
write('build/Abbe.tex', make_SI(abbe,r'',figures=0))



################################################################
### Auflösungsvermögen
################################################################
b = 0.03
A = b * (values[0]*wavelength**(-3)) / (nFkt(wavelength))
print('Auflösungsvermögen:', A)

write('build/Auflu.tex', make_table([wavelength*10**9, A],[1,0]))



################################################################
### Absorptionsstelle (n=1)
################################################################
wave1 = np.sqrt( values[0]/(values[1]-1) )
print('Absorptionsstelle:', wave1)

write('build/absorb.tex', make_SI(wave1*10**9,r'\nano\meter',figures=0))

