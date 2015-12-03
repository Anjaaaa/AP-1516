import numpy as np
from uncertainties import ufloat

v, v_fehler = np.genfromtxt('Geschwindigkeit_plot.txt', unpack = True)

f_00 = 20357.4
c = 360.1


for i in range(len(v)):
    f_e = f_00 * (1 + (v[i] / c))
    f_q = f_00 / (1 - (v[i] / c))
    diff = f_e - f_q
    print('Geschwindigkeit', i , v[i], f_e, f_q, 'Differenz', diff, '\n')
