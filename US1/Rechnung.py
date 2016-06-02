import numpy as np
from uncertainties import ufloat
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from table import(
	make_table,
	make_SI,
	write,
	make_composed_table)


h, r = np.genfromtxt('Messung_Zylindermasse.txt', unpack = True)
#in SI
h = h*10**-3
r = r*10**-3


################## Schallgeschwindigkeit nach Impuls echo Methode berechnen
laufzeit_1 = np.genfromtxt('Messung_1.txt', unpack = True)
hohe_1 = np.array([h[0], h[1], h[0]+h[1], h[2], h[3], h[0]+h[2]])
#in SI
laufzeit_1 = laufzeit_1*10**-6

write('build/tabelle_echo-impuls.txt', make_table([laufzeit_1*10**5, hohe_1*10**3], [2,2]))

def  g(l, m, b):
    return 0.5*l*m + b #0.5 da doppelte strecke bei reflektion zurückgelegt wird

parameters, popt = curve_fit(g, laufzeit_1, hohe_1)

m = ufloat(parameters[0], np.sqrt(popt[0,0])) #schallgeschwingigkeit in acryl in m/s
b = ufloat(parameters[1], np.sqrt(popt[1,1])) # dicke der kontaktschicht in m
write('build/v_Echo-Impuls.txt', make_SI(m, r'\meter\per\second', figures=1))
write('build/b_Echo-Impuls.txt', make_SI(b*10**3, r'\milli\meter', figures=2))
print(m)
print(b)

x = np.linspace(0, 10)
plt.plot(laufzeit_1*10**5, hohe_1, 'ro', label='Messdaten')
plt.plot(x, g(x, *parameters)*10**-5, 'k-', label='Ausgleichsgerade')
plt.xlabel('Laufzeit in $\mathrm{s}^{-5}$')
plt.ylabel('Höhe der Zylinder in m')
plt.legend(loc='best')
plt.savefig('build/Impuls-Echo.png')
plt.show()

############## Schallgeschwindigkeit nach der Durchschallungsmethode 
laufzeit_2 = np.genfromtxt('Messung_2.txt', unpack = True)
hohe_2 = np.array([h[0], h[1], h[2], h[3], h[0]+h[1]])
# in SI
laufzeit_2 = laufzeit_2*10**-6

write('build/tabelle_Durchschall.txt', make_table([laufzeit_1*10**5, hohe_1*10**3], [2,2]))

def  g_2(l, m, b):
    return l*m + b

parameters, popt = curve_fit(g_2, laufzeit_2, hohe_2)

m_2 = ufloat(parameters[0], np.sqrt(popt[0,0])) #schallgeschwingigkeit in acryl in m/s
b = ufloat(parameters[1], np.sqrt(popt[1,1])) # dicke der kontaktschicht in m
write('build/v_Durchschall.txt', make_SI(m_2, r'\meter\per\second', figures=1))
write('build/b_Durchschall.txt', make_SI(b*10**3, r'\milli\meter', figures=2))
print(m)
print(b)

x = np.linspace(min(laufzeit_2), max(laufzeit_2))
plt.plot(laufzeit_2, hohe_2, 'ro', label='Messdaten')
plt.plot(x, g_2(x, *parameters), 'k-', label='Ausgleichsgerade')
plt.xlabel('Laufzeit in $\mathrm{s}^{-5}$')
plt.ylabel('Höhe der Zylinder in m')
plt.legend(loc='best')
plt.savefig('build/Durchschall.png')
plt.show()


#############Schichtdicken Berechnen:
v_mittel = (m + m_2)/2 #Schallgeschwindigkeit aus beiden Aufgabenteilen mitteln
write('build/v_mittel.txt', make_SI(v_mittel, r'\meter\per\second', figures=1))

t_1 = (22.74 - 13.5)*10**-6
t_2 = (35.7 - 22.74)*10**-6

h_1 = 0.5*v_mittel*t_1
h_2 = 0.5*v_mittel*t_2
print('Dicke erste Schicht:', h_1)
print('Dicht zweite Schicht:', h_2)


##### Abmessungen des Auges:
sonde_iris = 0.5*1410*11.6*10**-6 #zeit bis zum ersten maximum
dicke_linse = 0.5*2500*(16.6-11.6)*10**-6 #zeit zwischen erstm und zweiten maximum
linse_retina = 0.5*1410*(73.9-16.6)*10**-6 #zeit zweischen zweiten und viertem Maximum, drittes ist doppelte reflexion an der linse

#gesamtlänge des auges
print(sonde_iris, dicke_linse, linse_retina)
auge = sonde_iris + dicke_linse + linse_retina


print(auge)

#s = ['1. Max', '2. Max', '3. Max']
#t = np.array([1,2,3])

#print(make_table([s, t], [0,2]))



#print(s)






