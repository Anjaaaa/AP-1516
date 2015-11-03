import numpy as np
from uncertainties import ufloat
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp



# h sind die Werte, wo sich die Quelle auf den Empfänger zu fährt.
# w sind die Werte, wo sich die Quelle vom Empfänger entfernt.


f_6h, f_12h, f_18h, f_24h, f_30h, f_36h, f_42h, f_48h, f_54h, f_60h = np.genfromtxt('Quelle_bewegt_sich_hin.txt', unpack = True )
f_hin = [f_6h, f_12h, f_18h, f_24h, f_30h, f_36h, f_42h, f_48h, f_54h, f_60h]


f_6w, f_12w, f_18w, f_24w, f_30w, f_36w, f_42w, f_48w, f_54w, f_60w = np.genfromtxt('Quelle_bewegt_sich_weg.txt', unpack = True )
f_weg = [f_6w, f_12w, f_18w, f_24w, f_30w, f_36w, f_42w, f_48w, f_54w, f_60w]




# Für den Auswertungsteil im Protokoll

for i in range(len(f_hin)):
    f_hin[i] = ufloat(np.mean(f_hin[i]), np.std(f_hin[i]/len(f_hin[i])))
print('f_hin:', f_hin)


for i in range(len(f_weg)):
    f_weg[i] = ufloat(np.mean(f_weg[i]), np.std(f_weg[i]/len(f_weg[i])))
print('f_weg:', f_weg)


