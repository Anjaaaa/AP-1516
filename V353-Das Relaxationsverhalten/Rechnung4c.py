import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties import ufloat

f, A, a = np.genfromtxt('4bc.txt', unpack = True)
a = a * 10**(-6) #mikrosekunde in sekunde
U_0 = 20


#Pasenverschiebung
phi = 2 * np.pi * a * f

def  g(f, c):
    return np.arctan(- 2 * np.pi * f * c)
# return np.arctan(- 2 * np.pi * f * c) + d funktioniert irgendiwe nicht

parameters, popt = curve_fit(g, f, phi)

c = ufloat(parameters[0], np.sqrt(popt[0,0]))
#d = ufloat(parameters[1], np.sqrt(popt[1,1]))

print('Zeitkonstante', c)
#print('Plotfehler', d)

print(popt)

x = np.linspace(0, 1400)
plt.plot(f, phi, 'rx')
plt.plot(x, g(x, *parameters), 'b-')
plt.ylabel('Winkel in Bogenmaß')
plt.xlabel('Frequenz / Hz')
# erste Liste: Tick-Positionen, zweite Liste: Tick-Beschriftung
plt.yticks([0, np.pi/8 , np.pi / 4, 3*np.pi /8, np.pi / 2],[r"$0$", r"$\frac{1}{8}\pi$", r"$\frac{1}{4}\pi$", r"$\frac{3}{8}\pi$", r"$\frac{1}{2}\pi$"])
plt.savefig('Phasenverschub1.png')
plt.show()

#Aufgabe 4d:
r = np.linspace(0, 10, 500)
theta = 2 * np.pi * r

def A2(phi):
    return 1 / np.sqrt(1+np.tan(phi2)**2)

phi2 = np.linspace(0, np.pi /2)
i = np.arange(0,len(phi), 2) #nur jeden zweiten wert plotten

plt.polar(phi2, A2(phi2))
plt.polar(phi[i], A[i]/U_0, 'rx')
plt.show()
