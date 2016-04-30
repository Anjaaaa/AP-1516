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



wavelength, phi_1, phi_2, eta_1, eta_2 = np.genfromtxt('Werte.txt', unpack = True)

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
eta = 180 - (360 + eta_1 - eta_2)
print('eta:', eta)
print('phi_Mittel:', phi_Mittel)


write('build/Messwerte1.tex', make_table([phi_1, phi_2, phi],[0,0,0]))
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
plt.plot(wavelength*10**9, n_nom, 'gx', label='Brechungsindex')
plt.plot(wavelength*10**9, eta/30, 'bx', label='Brechwinkel/30')


plt.xlabel(r'$\mathrm{\lambda} \ /\  \mathrm{nm}$')
plt.ylabel(r'$n$')
plt.legend(loc='best')
plt.show()
################################################################



################################################################
### Entscheidung für eine der beiden Gleichungen 11 bzw. 11a durch Regression
### WICHTIG! 1 hier ist gestrichen im Protokoll und 2 hier ist ungestrichen im Protokoll
################################################################
### wavelength^2 -> n^2
print('Regression mit Lambda^2')
# Hier mache ich die Wellenlänge größer, weil Python sonst ein Speicherproblem bekommt, weil die Zahlen so klein sind.
X = (wavelength*10**9)**2
regression1 = np.polyfit(X, Y, 1)
regFkt1 = np.poly1d(regression1)

values1, errors1 = curve_fit(f, X, Y)


steigung1 = 10**(-18)*ufloat(values1[0],np.sqrt(errors1[0,0]))
versch1 = ufloat(values1[1], np.sqrt(errors1[1,1]))
print('Steigung:', steigung1)
print('y-Achsenabschnitt:', versch1)

plt.plot(X, Y, 'r.', label='Datenpunkte')
plt.plot(X, regFkt1(X), 'g-', label = 'Regressionsgerade')

plt.xlabel(r'$\mathrm{\lambda^2} \ /\  \mathrm{m}^2$')
plt.ylabel(r'$n^2$')

write('build/Reg1Steig.tex', make_SI(steigung1*10**(25),r'\meter^{-2}','e-25',figures=2))
write('build/Reg1Versch.tex', make_SI(versch1,r'',figures=2))
write('build/Reg1SteigRel.tex', make_SI(unp.std_devs(steigung1)/steigung1.n,r'\meter^{-2}',figures=2))
write('build/Reg1VerschRel.tex', make_SI(unp.std_devs(versch1)/versch1.n,r'',figures=2))

plt.show()


### 1/wavelength^2 -> n^2
print('Regression mit 1/Lambda^2')
X = 1 / wavelength**2
regression2 = np.polyfit(X, Y, 1)
regFkt2 = np.poly1d(regression2)

values2, errors2 = curve_fit(f, X, Y)

steigung2 = ufloat(values2[0],np.sqrt(errors2[0,0]))
versch2 = ufloat(values2[1], np.sqrt(errors2[1,1]))
print('Steigung:', steigung2)  
print('y-Achsenabschnitt:', versch2)

plt.plot(X, Y, 'r.', label='Datenpunkte')
plt.plot(X, regFkt2(X), 'g-', label = 'Regressionsgerade')

plt.xlabel(r'$\mathrm{\frac{1}{\lambda^2}} \ /\  \mathrm{m}^{-2}$')
plt.ylabel(r'$n^2$')

write('build/Reg2Steig.tex', make_SI(steigung2*10**(14),r'\meter\squared','e-14',figures=2))
write('build/Reg2Versch.tex', make_SI(versch2,r'',figures=2))
write('build/Reg2SteigRel.tex', make_SI(-unp.std_devs(steigung2)/steigung2.n,r'\meter\squared',figures=2))
write('build/Reg2VerschRel.tex', make_SI(unp.std_devs(versch2)/versch2.n,r'',figures=2))

plt.show()



### Abweichungsquadrate
abw2 = 1/6 * sum((n_nom**2 - versch2.n - steigung2.n / wavelength**2)**2)
abw1 = 1/6 * sum((n_nom**2 - versch1.n - steigung1.n * wavelength**2)**2)
print('n_nom**2:', n_nom**2)
print('P_0:', versch2.n)
print('P_2:', steigung2.n/wavelength**2)
print('s**2:', abw2)
print('s**2Strich:', abw1)

write('build/abw1.tex', make_SI(abw1*1000,r'','e-3',figures=1))
write('build/abw2.tex', make_SI(abw2*1000,r'','e-3',figures=1))



################################################################
### Endgültige Dispersionskurve
################################################################
def nFkt(wavelength):
     return np.sqrt( values2[1]+values2[0]/wavelength**2 )

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

abbe = ( nFkt(D)-1 ) / ( nFkt(F)-nFkt(C) )
write('build/Abbe.tex', make_SI(abbe,r'',figures=3))



################################################################
### Auflösungsvermögen
################################################################
b = 0.02
A = - b * (values2[0]*wavelength**(-3)) / (nFkt(wavelength))
print(A)

write('build/Aufl.tex', make_table([wavelength*10**9, A],[1,0]))



################################################################
### Absorptionsstelle (n=1)
################################################################
wave1 = np.sqrt( -values2[0]/(values2[1]-1) )
print(wave1)

write('build/absorb.tex', make_SI(wave1*10**8,r'\nano\meter','e-8',figures=3))

