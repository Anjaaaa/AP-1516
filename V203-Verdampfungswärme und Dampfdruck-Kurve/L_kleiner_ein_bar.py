import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

T, P = np.genfromtxt('Messreihe1.txt', unpack = True)

T = 273.15 + T
P = 10*P #mbar in Pascal

T = 1/T
P = np.log(P)


fit = np.polyfit(T, P, 1)
fit_fn = np.poly1d(fit)

def f(T, a, b):
    return a*T +b

parameters, popt = curve_fit(f, T, P)

R = 8.314 #gaskonstante
L = - parameters[0] * R
print (L)


plt.plot(T, P, 'rx', label = 'Datenpunkte')
plt.plot(T, fit_fn(T), 'g-', label='Regressionsfunktion')
plt.xlabel(r'$1/T \ /\  (\mathrm{1/K})$')
plt.ylabel(r'$log(P)$')
plt.legend(loc='best')
plt.savefig('Regerssionspolynom_P(t).pdf')
plt.show()
