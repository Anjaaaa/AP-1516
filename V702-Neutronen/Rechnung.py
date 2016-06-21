import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import unumpy as unp
from uncertainties.umath import *
from table import(
        make_table,
        make_SI,
        make_full_table,
        write)

### Nullmessung Brom: 182 Pulse in 900s
### Nullmessung Silber: 222 Pulse in 1200s

OffSetBrom = 182/900
OffSetSilber = 222/1200

### Intervall Brom: 180s
### Intervall Silber: 10s
SilberGemessen, BromGemessen = np.genfromtxt('Messwerte.txt', unpack = True)

# Der radioaktive Zerfall ist eine Poissonverteilung. Der Fehler des Mittelwertes ist dabei die Wurzel des Mittelwertes. D wir jeden Wert nur einmal gemessen haben ist der Fehler eines Wertes x die Wurzel aus o.

BromGemessen = BromGemessen[np.invert(np.isnan(BromGemessen))]

Silber = unp.uarray(np.zeros(42), np.zeros(42))
Brom = unp.uarray(np.zeros(10), np.zeros(10))

for i in range(0,9):
   Brom[i] = ufloat(BromGemessen[i], np.sqrt(BromGemessen[i]))
   print(i)
   print(Brom[i])
for i in range(0,41):
   Silber[i] = ufloat(SilberGemessen[i], np.sqrt(SilberGemessen[i]))

tBrom = np.array([1,2,3,4,5,6,7,8,9,10])


plt.semilogy(tBrom*3, unp.nominal_values(Brom), 'ro', label = 'Messdaten')
plt.errorbar(tBrom*3, unp.nominal_values(Brom), xerr=0, yerr=unp.std_devs(Brom), fmt='ro')
plt.xticks(np.arange(0, 3*(max(tBrom)+1), 3.0))
plt.legend(loc='best')
plt.xlabel('Zeit $t$ in Minuten')
plt.ylabel('Anzahl der Pulse in $10s$ ')

plt.show()

write('build/austrittsarbeit.tex', make_SI(ufloat(parameters[1],np.sqrt(popt[1,1])), r'\volt', figures=2))
write('build/Winkel.tex', make_table([PhiHe, PhiNa, PhiKa, PhiRu],[1,1,1,1]))
write('build/WinkelGanz.tex', make_full_table(
    r'Gemessene mittlere Beugungswinkel $\varphi$ in \si{\degree} der Dubletts von Natrium, Kalium und Rubidium',
    'tab:Winkel',
    'build/Winkel.tex',
    [],
    [r'$\varphi_\text{\ce{He}}$',
    r'$\varphi_\text{\ce{Na}}$',
    r'$\varphi_\text{\ce{K}}$',
    r'$\varphi_\text{\ce{Ru}}$']))
