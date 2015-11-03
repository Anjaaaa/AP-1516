import numpy as np
from uncertainties import ufloat

t_6, t_12, t_18, t_24, t_30, t_36, t_42, t_48, t_54, t_60 = np.genfromtxt('Geschwindigkeit.txt', unpack = True)

t = [t_6, t_12, t_18, t_24, t_30, t_36, t_42, t_48, t_54, t_60]

s = 0.420     # Abstand zwischen den Lichtschranken



# Umrechnung von milli Sekunden in Sekunden

for i in range(len(t)):
    t[i] /= 1000



# Berechnung der Fehler der Zeitmessung

for i in range(len(t)):
    t[i] = ufloat(np.mean(t[i]), np.std(t[i])/np.sqrt(len(t[i])))

print('Fehler der Zeitmessung:',t)

# Berechnung der Geschwindigkeiten

for i in range(len(t)):
    v = s / t[i]
    a = i*6 + 6
    print(a, 'Umdrehungen:', v)

