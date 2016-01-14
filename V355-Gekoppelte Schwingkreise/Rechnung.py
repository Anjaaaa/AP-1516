import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp
from tabulate import tabulate

C_k, nu_1, nu_2 = np.genfromtxt('Umrechnung.txt', unpack = True)
t_1 = 1/nu_1
t_2 = 1/nu_2



# Steigung der Frequenz

nu_start = 11.16 * 1000   # Hz
nu_end = 112.6 * 1000     # Hz

T = 1     # Zeit in der die Spanne von nu_start bis nu_end durchlaufen wurde = 1s


m = (nu_end - nu_start) / T     # Steigung m von nu(t) = m * t + nu_start


def f(t, m, nu_start):
    return m*t+nu_start



#print('Leerlaufspannung Messung B:', parameters[1], '+/-', np.sqrt(popt[1,1]))
#print('Innenwiderstand Messung B:', parameters[0], '+/-', np.sqrt(popt[0,0]))
x = np.linspace(-0.1,1.1)
plt.plot(x, f(x, m, nu_start), 'b-')
plt.plot(0, nu_start, 'rx', label = 'Start bzw. Ende')
plt.plot(1, nu_end, 'rx')
#plt.xlabel('Stromstärke $\mathrm{I} \ /\  \mathrm{A}$')
#plt.ylabel('Spannung $\mathrm{U} \ /\  \mathrm{V}$')
plt.legend(loc='best')
#plt.savefig('Spannung_Messung_b.pdf')
plt.show()

print('Zeit nach der nu+ bzs nu- erreicht wurden:', C_k, t_1, t_2)
print('Frequenz bei t_1:', C_k, f(t_1, m, nu_start))
print('Frequenz bei t_2:', C_k, f(t_2, m, nu_start))


n_dyn = 0.5*(f(t_1, m, nu_start)+f(t_2, m, nu_start)) / (f(t_1, m, nu_start)-f(t_2, m, nu_start))
print('Schwingungsbauche dynmaisch:', n_dyn)

C_k, nu_plus, nu_minus = np.genfromtxt('Frequenz_Phase.txt', unpack = True)
nu_plus *= 1000
nu_minus *= 1000

n_phase = 0.5*(nu_plus + nu_minus) / (nu_plus - nu_minus)
print('Schwingungsbauche bestimmt uber die Phase:', n_phase)


n = np.genfromtxt('maxima.txt', unpack = True)

nu_res = 33.33 * 1000
nu_min = nu_res/(2*n) * (1+2*n)

print(nu_min)
