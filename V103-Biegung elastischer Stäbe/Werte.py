import numpy as np
from uncertainties import ufloat
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp


# Abstand von der Mitte = x
# Durchbiegung mit Gewicht = D_mit
# Durchbiegung ohne Gewicht = D_ohne

x, D_mit_links, D_mit_rechts, D_ohne_links, D_ohne_rechts = np.genfromtxt('Biegung_beidseitig_eingespannt.txt', unpack = True)

D_mit_links = D_mit_links/1000                        # von mm in m umrechnen
D_mit_rechts = D_mit_rechts/1000

D_ohne_links = D_ohne_links/1000                        # von mm in m umrechnen
D_ohne_rechts = D_ohne_rechts/1000


x = x/100   # von cm in m umrechnen

X_links =  0.2775 - x
X_rechts = X_links


D_ohne_l = ufloat(np.mean(D_ohne_links), np.std(D_ohne_links)/np.sqrt(len(D_ohne_links)))
print('Mittelwert D ohne Gewicht links:', D_ohne_l)

D_ohne_r = ufloat(np.mean(D_ohne_rechts), np.std(D_ohne_rechts)/np.sqrt(len(D_ohne_rechts)))
print('Mittelwert D ohne Gewicht rechts:', D_ohne_r)



plt.plot(X_links, D_mit_links, 'r.', label='Datenpunkte_links_mit')
plt.plot(X_rechts, D_mit_rechts, 'g.', label = 'Datenpunkte_rechts_mit')
plt.xlabel(r'$x \ /\  \mathrm{m}$')
plt.ylabel(r'$D \ /\  \mathrm{m}$')
plt.legend(loc='best')
plt.savefig('Werte_mit')
plt.show()



plt.plot(X_links, D_ohne_links, 'r.', label='Datenpunkte_links_ohne')
plt.plot(X_rechts, D_ohne_rechts, 'g.', label = 'Datenpunkte_rechts_ohne')
plt.xlabel(r'$x \ /\  \mathrm{m}$')
plt.ylabel(r'$D \ /\  \mathrm{m}$')
plt.legend(loc='best')
plt.savefig('Werte_ohne')
plt.show()



D_links = D_ohne_links - D_mit_links          # Differenz berechnen
D_rechts = D_ohne_rechts - D_mit_rechts

D = (D_links + D_rechts) / 2


fit = np.polyfit(X_links, D, 1)
fit_fn = np.poly1d(fit)

def f(X_links, m, b):
    return m*X_links+b

parameters, popt = curve_fit(f, X_links, D)


print('Steigung:', parameters[0])
print('Fehler Steigung:', np.sqrt(popt[0,0]))
print('y-Achsenabschnitt:', parameters[1])
print('Fehler y-Achsenabschnitt:', np.sqrt(popt[1,1]))
print('Elastizitätsmodul:', 1/parameters[0])



plt.plot(X_links, fit_fn(X_links), 'y-', label = 'Regressionsgerade')
plt.plot(X_links, D_links, 'r.', label='Datenpunkte_links')
plt.plot(X_rechts, D_rechts, 'g.', label = 'Datenpunkte_rechts')
plt.plot(X_links, D, 'b.', label='Datenpunkte_gemittelt_links_und_rechts')
plt.xlabel(r'$x \ /\  \mathrm{m}$')
plt.ylabel(r'$D \ /\  \mathrm{m}$')
plt.legend(loc='best')
plt.savefig('Wertedifferenz')
plt.show()


