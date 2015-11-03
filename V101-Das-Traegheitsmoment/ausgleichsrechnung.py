import numpy as np

# Ausgleichsrechnung
A, T = np.genfromtxt('eigentragheitsmoment_Abstand und Winkel.txt', unpack = True)
T = T**2
A = (A/1000)**2
T = T[np.invert(np.isnan(T))]
A = A[np.invert(np.isnan(A))]


T_av = np.average(T)
A_av = np.average(A)

Sxy = np.sum((A-A_av)*(T-T_av))
Sx2 = np.sum((A-A_av)**2)

m = Sxy/Sx2

print('Steigung: ', m)


