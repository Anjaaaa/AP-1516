import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp


Abstand, int, gain = np.genfromtxt('Werte_Lampe.txt', unpack = True)


# Gain herausrechnen
int = int / gain
int = int / 200

Abstand = 200 - Abstand
Abstand = Abstand / 100

print('Abstand:', Abstand)
print('Integral:', int)

def f(x, m, b):
    return m*x + b

parameters, popt = curve_fit(f, np.log(Abstand), np.log(int))
fit = np.polyfit(np.log(Abstand), np.log(int), 1)
regression = np.poly1d(fit)

plt.plot(np.log(Abstand), np.log(int), 'rx', label = 'Messwerte')
x = np.linspace(-2, 1)
plt.plot(x, regression(x), 'b-', label = 'Regression')
plt.xlabel('Abstand $\mathrm{\ln(r/m)}$')
plt.ylabel('Spannung $\mathrm{\ln(U/V)}$')
plt.legend(loc='best')
plt.savefig("Regression.png")
plt.show()



print('Steigung:', parameters[0], np.sqrt(popt[0,0]))
print('y-Achsenabschnitt:', parameters[1], np.sqrt(popt[1,1]))

r = np.linspace(0.14, 2)
plt.plot(Abstand, 1000*int, 'rx', label = 'Messwerte')
plt.plot(r, 1000*np.exp(parameters[1])*r**parameters[0], 'b-', label = '$\mathrm{U(r)}$')
plt.xlabel('Abstand $\mathrm{r \ / \ m}$')
plt.ylabel('Spannung $\mathrm{U \ /\ mV}$')
plt.legend(loc='best')

plt.savefig("U(r).png")
plt.show()

