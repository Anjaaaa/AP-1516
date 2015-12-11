import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

U = 1
R_i = 15

x = np.linspace(0,100)
plt.plot(x, U**2/(R_i+x)**2*x, 'b-', label='Leistung bei $R_i=15\Omega$ und $U=1$V')
plt.axvline(x=15, color='r', linestyle='-',label='$R_{a,max} = R_i$')
plt.xlabel('Widerstand')
plt.ylabel('Leistung')
plt.legend(loc='best')
plt.savefig('Leistung.pdf')
plt.show()

