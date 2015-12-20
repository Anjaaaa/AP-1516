import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp


phi_ohne, int_ohne = np.genfromtxt('Phase_Int_ohne.txt', unpack = True)
phi_mit, int_mit = np.genfromtxt('Phase_Int_mit.txt', unpack = True)


# Der Gain war eigentlich 10*20*20, der hier fehlende Faktor 1000 ist in der Legende mit kilo Volt.
int_ohne = int_ohne * 4
int_mit = int_mit * 4


# Bogenmaß
phi_ohne = phi_ohne * 2*np.pi / 360
phi_mit = phi_mit * 2*np.pi / 360






def cosinus(phi, A, w, phi_0):
    return A*np.cos(w*phi+phi_0)



##############################################################################################################
#   Fit für die Werte ohne das Rauschen   ####################################################################

parameters_ohne, popt_ohne = curve_fit(cosinus, phi_ohne, int_ohne)



print('Streckung in y-Richtung bei ohne:', parameters_ohne[0])
print('Streckung in x-Richtung bei ohne:', parameters_ohne[1])
print('popt:', parameters_ohne)
print('Verschiebung in x-Richtung bei ohne:', parameters_ohne[2] / np.pi, 'pi')
print('Fehler Streckung y ohne (schon gewurzelt):', np.sqrt(popt_ohne[0,0]))
print('Fehler Streckung x ohne (schon gewurzelt):', np.sqrt(popt_ohne[1,1]), '\n')



phi = np.linspace(-np.pi/8, 17/8 * np.pi, 1000)
plt.xlim(-np.pi/8, 17/8 *np.pi)
plt.xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
           [r"$0$", r"$\frac{1}{2}\pi$", r"$\pi$", r"$\frac{3}{2}\pi$", r"$2\pi$"])


plt.plot(phi, cosinus(phi, parameters_ohne[0], parameters_ohne[1], parameters_ohne[2]), 'b-', label = 'FIT')
plt.plot(phi_ohne, int_ohne, 'rx', label = 'Messwerte')
plt.plot(phi, cosinus(phi, 30, 1, 0), 'g-', label = 'Theorie')


plt.xlabel('Phase $\mathrm{\phi}$')
plt.ylabel('Spannung $\mathrm{U} \ /\  \mathrm{kV}$')
plt.legend(loc='best')

plt.show()



##############################################################################################################
#   Fit für die Werte mit Rauschen   #########################################################################


parameters_mit, popt_mit = curve_fit(cosinus, phi_mit, int_mit)



print('Streckung in y-Richtung bei mit:', parameters_mit[0])
print('Streckung in x-Richtung bei mit:', parameters_mit[1])
print('popt:', parameters_mit)
print('Verschiebung in x-Richtung bei mit:', parameters_mit[2] / np.pi, 'pi')
print('Fehler Streckung y mit (schon gewurzelt):', np.sqrt(popt_mit[0,0]))
print('Fehler Streckung x mit (schon gewurzelt):', np.sqrt(popt_mit[1,1]), '\n')



phi = np.linspace(-np.pi/8, 17/8 * np.pi, 1000)
plt.xlim(-np.pi/8, 17/8 * np.pi)
plt.xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
           [r"$0$", r"$\frac{1}{2}\pi$", r"$\pi$", r"$\frac{3}{2}\pi$", r"$2\pi$"])


plt.plot(phi, cosinus(phi, parameters_mit[0], parameters_mit[1], parameters_mit[2]), 'b-', label = 'FIT')
plt.plot(phi_mit, int_mit, 'rx', label = 'Messwerte')
plt.plot(phi, cosinus(phi, 30, 1, 0), 'g-', label = 'Theorie')


plt.xlabel('Phase $\mathrm{\phi}$')
plt.ylabel('Spannung $\mathrm{U} \ /\  \mathrm{kV}$')
plt.legend(loc='best')



plt.show()
