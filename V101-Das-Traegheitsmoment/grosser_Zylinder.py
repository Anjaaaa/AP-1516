import numpy as np
from uncertainties import ufloat

h, d, T, gewicht = np.genfromtxt('grosser_Zylinder.txt', unpack = True)

r = d/2
T = T[np.invert(np.isnan(T))]
r = r[np.invert(np.isnan(r))] /1000
h = h[np.invert(np.isnan(h))] /1000
gewicht = gewicht[np.invert(np.isnan(gewicht))] /1000



T_gesamt = ufloat(np.mean(T), np.std(T)/np.sqrt(len(T)))
r_gesamt = ufloat(np.mean(r), np.std(r)/np.sqrt(len(r)))
h_gesamt = ufloat (np.mean(h), np.std(h)/np.sqrt(len(h)))

I_theoretisch = gewicht*(r_gesamt**2/4+h_gesamt**2/12)


I_experimentell = T_gesamt**2/(4*np.pi**2)*0.02857

I_D = I_experimentell - I_theoretisch

print('theoretisch: ', I_theoretisch)
print('experimentell: ', I_experimentell)
print('Eigenträgheitsmoment: ', I_D)
print('Schwingungsdauer:', T_gesamt)
print('Höhe, Radius:', h_gesamt, r_gesamt)
