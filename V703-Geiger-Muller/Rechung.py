import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from table import (
    make_table,
    make_full_table,
    make_SI,
    write,)

U, c = np.genfromtxt('Messung1.txt', unpack = True)

t = 10 #Messzeit 10 sekunden
Z = c / t #Zählrate in counts/second
Z_fehler = np.sqrt(c)/t

write('build/tabelle_charakteristik.txt', make_table([U[:15], Z[:15], Z_fehler[:15], U[15:], Z[15:], Z_fehler[15:]], [2,2,2,2,2,2]))

plt.plot(U, Z, 'ko', label='Messwerte')
plt.errorbar(U, Z, xerr=1, yerr=Z_fehler, fmt='r.', label = r'Fehler $\sqrt{n}$')
plt.legend(loc='best')
plt.xlabel(r'Spannung $U \ /\  \mathrm{V}$') 
plt.ylabel(r'Zählrate $Z \ /\ {\mathrm{Counts}}/{\mathrm{s}}$')
plt.xlim(0,750)
plt.savefig('build/charakteristik_gesamt.png')
plt.show()


def linear(x, m, b):
	return m*x+b
	


parameters1, popt1 = curve_fit(linear, U[15:27], Z[15:27])
m1 = ufloat(parameters1[0], np.sqrt(popt1[0,0]))
b1 = ufloat(parameters1[1], np.sqrt(popt1[1,1]))

write('build/m1.txt', make_SI(m1, r'\per\volt\per\second', figures=1))
write('build/b1.txt', make_SI(b1, r'\per\second', figures=1))



x = np.linspace(300,750)
plt.plot(U[12:], Z[12:], 'ro', label='Messwerte')
plt.plot(U[15:27], Z[15:27], 'ko', label='Messwerte (linearer Teil)')
plt.plot(x, linear(x, *parameters1),'k-', label = 'Regression (linearer Teil)')
plt.errorbar(U[12:], Z[12:], xerr=1, yerr=Z_fehler[12:], fmt='r.', label = r'Fehler $\sqrt{n}$')
plt.legend(loc='best')
plt.xlabel(r'Spannung $U \ /\  \mathrm{V}$') 
plt.ylabel(r'Zählrate $Z \ /\ {\mathrm{Counts}}/{\mathrm{s}}$')
plt.savefig('build/charakteristik_linear.png')
plt.show()

#######  steigung in %/100V herausfinden nach ebberg-methode (siehe mail)

#bestimme das Mittlere U_A: 
U_mittel = (U[27]+U[15])/2

#bestimme N_mittel, N1 und N2:
Z1 = m1*(U_mittel-50)+b1
Z2 = m1*(U_mittel+50)+b1
Z_mittel = m1*(U_mittel)+b1

# steigung nach ebberg
s = 100 * (Z2 - Z1)/Z_mittel #s in %/100V

print(Z1, Z_mittel, Z2, s, len(U))

write('build/U_mittel.txt', make_SI(U_mittel, r'\volt', figures=1))
write('build/U1.txt', make_SI(U_mittel-50, r'\volt', figures=1))
write('build/U2.txt', make_SI(U_mittel+50, r'\volt', figures=1))
write('build/Z_mittel.txt', make_SI(Z_mittel, r'\per\second', figures=1))
write('build/Z1.txt', make_SI(Z1, r'\per\second', figures=1))
write('build/Z2.txt', make_SI(Z2, r'\per\second', figures=1))

write('build/s.txt', make_SI(s, r'', figures=1))
#write('build/tabelle_messung1.txt', make_table([p1, puls1, x1], [0,0,2]))




