import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties import ufloat

f, A, a = np.genfromtxt('4bc.txt', unpack = True)
a = a * 10**(-6) #mikrosekunde in sekunde


def  g(f, b, c, d):
    return b / (np.sqrt(1+(2*np.pi*f)**2*c)) +d

parameter, popt = curve_fit(g, f, A)

b = ufloat(parameter[0], np.sqrt(popt[0,0]))
c = ufloat(parameter[1], np.sqrt(popt[1,1]))
d = ufloat(parameter[2], np.sqrt(popt[2,2]))

print('Zeitkonstante', unp.sqrt(c))
print('Ausgangsspannung', b)
print('Plotfehler', d)

print(popt)


x = np.linspace(0,1400)
plt.plot(f, A, 'rx')
plt.plot(x, g(x, *parameter), 'b-')
plt.ylabel('Amplitude / V')
plt.xlabel('Frequenz / Hz')
plt.savefig('Amplitude.png')
plt.show()
