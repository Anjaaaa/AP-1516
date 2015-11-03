import numpy as np
from uncertainties import ufloat
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp



# h sind die Werte, wo sich die Quelle auf den Empfänger zu fährt.
# w sind die Werte, wo sich die Quelle vom Empfänger entfernt.


f_6h, f_12h, f_18h, f_24h, f_30h, f_36h, f_42h, f_48h, f_54h, f_60h = np.genfromtxt('Quelle_bewegt_sich_hin.txt', unpack = True)
f_hin = [f_6h, f_12h, f_18h, f_24h, f_30h, f_36h, f_42h, f_48h, f_54h, f_60h]


f_6w, f_12w, f_18w, f_24w, f_30w, f_36w, f_42w, f_48w, f_54w, f_60w = np.genfromtxt('Quelle_bewegt_sich_weg.txt', unpack = True)
f_weg = [f_6w, f_12w, f_18w, f_24w, f_30w, f_36w, f_42w, f_48w, f_54w, f_60w]



# Frequenzen mit Fehler berechnen

for i in range(len(f_hin)):
    f_hin[i] = np.mean(f_hin[i])
print('f_hin:', f_hin)


for i in range(len(f_weg)):
    f_weg[i] = np.mean(f_weg[i])
print('f_weg:', f_weg)



# Frequenzunterschied berechnen

f_0 = np.genfromtxt('Ruhefrequenz.txt', unpack = True)
f_0 = np.mean(f_0)



# Array mit Ruhefrequenz erstellen

f_00 = np.zeros(10)
for i in range(len(f_00)):
    f_00[i] = f_0

print('Ruhefrequenz:', f_00)



# Frequenzunterschied berechnen

delta_f_hin = f_hin - f_00
print('delta_f_hin:', delta_f_hin)

delta_f_weg = f_weg - f_00
print('delta_f_weg:', delta_f_weg)



# Plotten

v, fehler = np.genfromtxt('Geschwindigkeit_plot.txt', unpack = True)

print('Geschwindigkeit:', v)



fit = np.polyfit(v, delta_f_hin, 1)
fit_fn = np.poly1d(fit)

def f(v, m, b):
    return m*v+b

parameters_h, popt_h = curve_fit(f, v, delta_f_hin)
print('Steigung bei hin:', parameters_h[0])
print('y-Achsenabschnit bei hin:', parameters_h[1])
print('Fehler Steigung hin (schon gewurzelt):', np.sqrt(popt_h[1,1]))
print('Fehler y-Achsenabschnitt hin (schon gewurzelt):', np.sqrt(popt_h[0,0]))
plt.plot(v, delta_f_hin, 'r.')
plt.plot(v, fit_fn(v), 'g-')
plt.show()


fit = np.polyfit(v, delta_f_weg, 1)
fit_fn = np.poly1d(fit)

def f(v, m, b):
    return m*v+b

parameters_w, popt_w = curve_fit(f, v, delta_f_weg)
print('Steigung bei f_weg:', parameters_w[0])
print('y-Achsenabschnitt bei weg:', parameters_w[1])
print('Fehler Steigung weg (schon gewurzelt):', np.sqrt(popt_w[1,1]))
print('Fehler y-Achsenabschnitt (schon gewurzelt):', np.sqrt(popt_w[0,0]))
plt.plot(v, delta_f_weg, 'r.')
plt.plot(v, fit_fn(v), 'g-')
plt.show()



Steigung_hin = ufloat(parameters_h[0], np.sqrt(popt_h[1,1]))
Inverses_Steigung_hin = 1/Steigung_hin
print('Inverses der Steigung hin (=lambda):', Inverses_Steigung_hin)


Steigung_weg = ufloat(parameters_w[0], np.sqrt(popt_w[1,1]))
Inverses_Steigung_weg = 1/Steigung_weg
print('Inverses der Steigung weg (=lambda):', Inverses_Steigung_weg)
