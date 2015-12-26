import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp


phi_ohne, int_ohne = np.genfromtxt('Phase_Int_ohne.txt', unpack = True)
phi_mit, int_mit = np.genfromtxt('Phase_Int_mit.txt', unpack = True)


# Gain 20*20*10
int_ohne = int_ohne / 4000
int_mit = int_mit / 4000


# Bogenmaß
phi_ohne = phi_ohne * 2*np.pi / 360
phi_mit = phi_mit * 2*np.pi / 360






def cosinus(phi, A, w, phi_0):
    return A*np.cos(w*phi + phi_0)



##############################################################################################################
#   Fit für die Werte ohne das Rauschen   ####################################################################

parameters_ohne, popt_ohne = curve_fit(cosinus, phi_ohne, int_ohne)



print('Streckung in y-Richtung bei ohne:', parameters_ohne[0])
print('Streckung in x-Richtung bei ohne:', parameters_ohne[1])
print('popt:', parameters_ohne)
print('Verschiebung in x-Richtung bei ohne:', parameters_ohne[2]/np.pi, 'pi')
print('Fehler Streckung y ohne (schon gewurzelt):', np.sqrt(popt_ohne[0,0]))
print('Fehler Streckung x ohne (schon gewurzelt):', np.sqrt(popt_ohne[1,1]))
print('Fehler Verschiebung ohne:', np.sqrt(popt_ohne[2,2])/np.pi, 'pi', '\n')


phi = np.linspace(-np.pi/8, 17/8 * np.pi, 1000)
plt.xlim(-np.pi/8, 17/8 *np.pi)
plt.xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
           [r"$0$", r"$\frac{1}{2}\pi$", r"$\pi$", r"$\frac{3}{2}\pi$", r"$2\pi$"])


plt.plot(phi, cosinus(phi, parameters_ohne[0], parameters_ohne[1], parameters_ohne[2]), 'b-', label = 'FIT')
plt.plot(phi_ohne, int_ohne, 'rx', label = 'Messwerte')
plt.plot(phi, cosinus(phi, 0.010*2/np.pi, 1, 0), 'g-', label = 'Theorie')


plt.xlabel('Phase $\mathrm{\phi}$')
plt.ylabel('Spannung $\mathrm{U} \ /\  \mathrm{V}$')
plt.legend(loc='best')


plt.savefig('Phase_ohne.png')
plt.show()



##############################################################################################################
#   Fit für die Werte mit Rauschen   #########################################################################


parameters_mit, popt_mit = curve_fit(cosinus, phi_mit, int_mit)



print('Streckung in y-Richtung bei mit:', parameters_mit[0])
print('Streckung in x-Richtung bei mit:', parameters_mit[1])
print('popt:', parameters_mit)
print('Verschiebung in x-Richtung bei mit:', (parameters_mit[2] )/np.pi, 'pi')
print('Fehler Streckung y mit (schon gewurzelt):', np.sqrt(popt_mit[0,0]))
print('Fehler Streckung x mit (schon gewurzelt):', np.sqrt(popt_mit[1,1]))
print('Fehler Verschiebung mit:', np.sqrt(popt_mit[2,2])/np.pi, 'pi', '\n')



phi = np.linspace(-np.pi/8, 17/8 * np.pi, 1000)
plt.xlim(-np.pi/8, 17/8 * np.pi)
plt.xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
           [r"$0$", r"$\frac{1}{2}\pi$", r"$\pi$", r"$\frac{3}{2}\pi$", r"$2\pi$"])


plt.plot(phi, cosinus(phi, parameters_mit[0], parameters_mit[1], parameters_mit[2]), 'b-', label = 'FIT')
plt.plot(phi_mit, int_mit, 'rx', label = 'Messwerte')
plt.plot(phi, cosinus(phi, 0.010*2/np.pi, 1, 0), 'g-', label = 'Theorie')


plt.xlabel('Phase $\mathrm{\phi}$')
plt.ylabel('Spannung $\mathrm{U} \ /\  \mathrm{V}$')
plt.legend(loc='best')


plt.savefig('Phase_mit.png')
plt.show()
