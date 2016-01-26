import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp
from tabulate import tabulate



# dynamische Messung

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

print('Zeit nach der nu+ bzs nu- erreicht wurden:', C_k, t_1, t_2)
print('Frequenz bei t_1:', C_k, f(t_1, m, nu_start))
print('Frequenz bei t_2:', C_k, f(t_2, m, nu_start))

nu_plus_dyn = ufloat(np.mean(f(t_1, m, nu_start)), np.std(f(t_1, m, nu_start))/np.sqrt(len(f(t_1, m, nu_start))))
print('Mittelwert nu_plus dyn Messung:', nu_plus_dyn)



# Schwingungen pro Bauch

def n(nu_pl, nu_mi):
    return 0.5*(nu_pl+nu_mi)/(nu_pl-nu_mi)


n_dyn = n(f(t_1, m, nu_start),f(t_2, m, nu_start))
print('Schwingungsbauche dynmaisch:', n_dyn)




# Messung über Phasenbeziehung

C_k, nu_plus, nu_minus = np.genfromtxt('Frequenz_Phase.txt', unpack = True)
nu_plus *= 1000
nu_minus *= 1000


print('nu_plus gemessen über Phase:', nu_plus)
print('nu_minus gemessen über Phase:', nu_minus)

n_phase = n(nu_plus, nu_minus)
print('Schwingungsbauche bestimmt uber die Phase:', n_phase)

nu_plus_phase = ufloat(np.mean(nu_plus), np.std(nu_plus)/np.sqrt(len(nu_plus)))
print('Mittelwert nu_plus über Phase bestimmt:', nu_plus_phase)




# Zählung der Schwingungen

n_gez = np.genfromtxt('maxima.txt', unpack = True)

nu_res = 33.33 * 1000
nu_min = nu_res/(2*n_gez) * (1+2*n_gez)

print('nu_minus berechnet aus den Schwingungen pro Bauch:', nu_min)






# Theoretische Werte

L = 23.954 / 1000
C_p = ( 0.7932 + 0.028 ) / 1000000000
C_k /= 1000000000
C_m = (C_k * 0.7932/1000000000)/  (2*0.7932/1000000000+C_k)   + 0.028/1000000000

print('C:', C_m)
nu_p = 1 / (2*np.pi*np.sqrt(L*C_p))
print('Erwartete Frequenz +:', nu_p)

nu_m = 1 / (2*np.pi*np.sqrt(L * C_m))
print('Erwartete Frequenz -:', nu_m)


n_e = n(nu_p,nu_m)

print('Erwartete Anzahl von Schwingungen:', n_e)



abw_dyn_minus = f(t_2, m, nu_start) / nu_m -1
abw_phase_minus = nu_minus / nu_m -1

print('Abweichung dyn nu_minus:', abw_dyn_minus)
print('Abweichung phase nu_minus:', abw_phase_minus)
