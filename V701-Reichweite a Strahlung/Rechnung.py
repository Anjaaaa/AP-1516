import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
from scipy.special import factorial
from uncertainties import ufloat
from table import (
    make_table,
    make_full_table,
    make_SI,
    write,)

p1, puls1, c1 = np.genfromtxt('Messung1.txt', unpack=True)
p2, puls2, c2 = np.genfromtxt('Messung2.txt', unpack=True)


##########################Bestimmung der Mittleren Reichweite von $\alpha$-Strahlung

#effektive wegstrecke x berechnen
p0 = 1013 #Normaldruck in mbar
a1 = 2.1 #unkorrigierte wegstecke in cm
a2 = 3 #unkorrigierte wegstrecke in cm
x1 = a1*p1/p0
x2 = a2*p2/p0

#lineare regression, um mittlere reichweite zu bestimmen

def linear(x, m, b):
	return m*x+b
	


parameters1, popt1 = curve_fit(linear, x1[16:], puls1[16:])
m1 = ufloat(parameters1[0], np.sqrt(popt1[0,0]))
b1 = ufloat(parameters1[1], np.sqrt(popt1[1,1]))

x = np.linspace(0,2.5)
plt.gca().yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x,_:x*10**(-3)))
plt.plot(x1, puls1, 'ro', label='Messdaten')
plt.plot(x, linear(x, *parameters1),'k-', label = 'Regression an den linearen Teil')
plt.xlabel('Effektiver Abstand zwischen Detektor und Strahler x in cm')
plt.ylabel('$10^3$ Pulse pro 120s')
plt.ylim(30000,120000)
plt.xlim(0,2.5)
plt.legend(loc='best')
plt.savefig('build/pulse1.png')
plt.show()

#mittelere reichweite bestimmen (lineare gleichung auflösen)
reichweite1 = (max(puls1)/2 - b1)/m1


write('build/reichweite1.txt', make_SI(reichweite1, r'\centi\meter', figures=2))
write('build/m1.txt', make_SI(m1, r'\per\centi\meter', figures=1))
write('build/b1.txt', make_SI(b1, r'', figures=1))
write('build/tabelle_messung1.txt', make_table([p1, puls1, x1], [0,0,2]))


########gleiches für messung2

parameters2, popt2 = curve_fit(linear, x2[12:], puls2[12:])
m2 = ufloat(parameters2[0], np.sqrt(popt2[0,0]))
b2 = ufloat(parameters2[1], np.sqrt(popt2[1,1]))

x = np.linspace(0,3)
plt.gca().yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x,_:x*10**(-3)))
plt.plot(x2, puls2, 'ro', label='Messdaten')
plt.plot(x, linear(x, *parameters2),'k-', label = 'Regression an den linearen Teil')
plt.xlabel('Effektiver Abstand zwischen Detektor und Strahler x in cm')
plt.ylabel('$10^3$ Pulse pro 120s')
plt.ylim(0,50000)
plt.xlim(0,3)
plt.legend(loc='best')
plt.savefig('build/pulse2.png')
plt.show()

#mittelere reichweite bestimmen (lineare gleichung auflösen)
reichweite2 = (max(puls2)/2 - b2)/m2
print(reichweite2)

write('build/reichweite2.txt', make_SI(reichweite2, r'\centi\meter', figures=2))
write('build/m2.txt', make_SI(m2, r'\per\centi\meter', figures=1))
write('build/b2.txt', make_SI(b2, r'', figures=1))
write('build/tabelle_messung2.txt', make_table([p2, puls2, x2], [0,0,2]))


##########################Bestimmung der Energie der $\alpha$-Strahlung

E1_emp = (reichweite1/0.31)**(2/3)
E2_emp = (reichweite2/0.31)**(2/3)


write('build/E1_emp.txt', make_SI(E1_emp, r'\mega\electronvolt', figures=1))
write('build/E2_emp.txt', make_SI(E2_emp, r'\mega\electronvolt', figures=1))


#bestimmen und energien
energie1 = c1/c1[0]*4 # energie in MeV
energie2 = c2/c2[0]*4 # energie in MeV
write('build/tabelle_energie1.txt', make_table([c1, energie1, x1], [0,3,2]))
write('build/tabelle_energie2.txt', make_table([c2, energie2, x2], [0,3,2]))

#regression an die energien zur bestimmung der steigung
x = np.linspace(0,2.2)

parameterse1, popte1 = curve_fit(linear, x1, energie1)
m_e1 = ufloat(parameterse1[0], np.sqrt(popte1[0,0]))
b_e1 = ufloat(parameterse1[1], np.sqrt(popte1[1,1]))

plt.plot(x1, energie1, 'ro', label='Messdaten')
plt.plot(x, linear(x, *parameterse1),'k-', label = 'Lineare Regression')
plt.xlim(0,2.2)
plt.legend(loc='best')
plt.savefig('build/energie1.png')
plt.show()


#messung2

parameterse2, popte2 = curve_fit(linear, x2[:13], energie2[:13])
m_e2 = ufloat(parameterse2[0], np.sqrt(popte2[0,0]))
b_e2 = ufloat(parameterse2[1], np.sqrt(popte2[1,1]))

plt.plot(x2, energie2, 'ro', label='Messdaten')
plt.plot(x, linear(x, *parameterse2),'k-', label = 'Regression an den linearen Teil')
plt.xlim(0,2.2)
plt.legend(loc='best')
plt.savefig('build/energie2.png')
plt.show()

#Energien berechnen
E1_reg = - m_e1 * reichweite1
E2_reg = - m_e2 * reichweite2

write('build/E1_reg.txt', make_SI(E1_reg, r'\mega\electronvolt', figures=1))
write('build/E2_reg.txt', make_SI(E2_reg, r'\mega\electronvolt', figures=1))
write('build/m_e1.txt', make_SI(m_e1, r'\mega\electronvolt\per\centi\meter', figures=1))
write('build/m_e2.txt', make_SI(m_e2, r'\mega\electronvolt\per\centi\meter', figures=1))


print(E1_emp, E1_reg)
print(E2_emp, E2_reg)
print(m_e1, m_e2)




################Statistik des radioaktiven Zerfalls

messung3 = np.genfromtxt('Messung3.txt', unpack=True)

#Mittelwert und Standartabweichung
messung3_mittel = np.mean(messung3)
messung3_std = np.std(messung3, ddof=1)

print('Mittelwert', messung3_mittel, 'Standtartabweichung', messung3_std)


# Histogramm zeichnen und Parameter bestimmen (n=Wahrscheinlichkeiten, bins = Intervallgrenzen)
n, bins, patches = plt.hist(messung3, bins=8, normed=1, facecolor='green')

#datensatz für poisson:
xp = np.linspace(1,8,8)

#datensatz für gauß:
xg = np.ones(len(bins)-1)
for i in range(0,len(bins)-1):
	xg[i] = (bins[i]+bins[i+1])/2
	
# regressionen (gauß und poisson)

def poisson(k, z):
	return z**k/factorial(k)*np.exp(-z)
	
def gauss(xp, s, mu):
	return 1/(s * np.sqrt(2*np.pi))*np.exp(-1/2*((xp-mu)/s)**2)
	

parameters_poisson, popt_poisson = curve_fit(poisson, xp, n, p0 = [np.mean(n)])

parameters_gauss, popt_gauss = curve_fit(gauss, xg, n, p0 = [np.mean(n), np.std(n)])

z = ufloat(parameters_poisson[0], np.sqrt(popt_poisson[0,0]))
s = ufloat(parameters_gauss[0], np.sqrt(popt_gauss[0,0]))
mu = ufloat(parameters_gauss[1], np.sqrt(popt_gauss[1,1]))
print(z, s, mu)


#plt.plot(xg, poisson(xp, *parameters_poisson), 'b-')

#plt.plot(xg, gauss(xg, *parameters_gauss), 'g-')

plt.plot(xg, n, 'ro')

#y = mlab.normpdf(bins, np.mean(n), np.std(n))

#plt.plot(bins, y, 'b--', linewidth=1)




#gaussglocke plotten
#x = np.linspace(3400, 4100)
#plt.plot(x, 1/(messung3_std * np.sqrt(2 * np.pi)) *np.exp( - (x - messung3_mittel)**2 / (2 * messung3_std**2) ),linewidth=2, color='r', label='Gaussverteilung')

#numpy.random.normal(loc=0.0, scale=1.0, size=None)

plt.legend(loc='best')
plt.show()


plt.plot(xg, gauss(xg, *parameters_gauss), 'g-')
plt.show()