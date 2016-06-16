import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats
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
plt.plot(x1[16:], puls1[16:], 'ro', label='Messdaten (Regression)')
plt.plot(x1[:15], puls1[:15], 'ko', label='Messdaten')
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
plt.plot(x2[12:], puls2[12:], 'ro', label='Messdaten (Regression')
plt.plot(x2[:11], puls2[:11], 'ko', label='Messdaten')
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

plt.plot(x1, energie1, 'ro', label='Messdaten (Regression)')
plt.plot(x, linear(x, *parameterse1),'k-', label = 'Lineare Regression')
plt.xlim(0,2.2)
plt.legend(loc='best')
plt.xlabel('Effektiver Abstand zwischen Detektor und Strahler x in cm')
plt.ylabel('Energie in MeV')
plt.savefig('build/energie1.png')
plt.show()


#messung2

parameterse2, popte2 = curve_fit(linear, x2[:13], energie2[:13])
m_e2 = ufloat(parameterse2[0], np.sqrt(popte2[0,0]))
b_e2 = ufloat(parameterse2[1], np.sqrt(popte2[1,1]))

plt.plot(x2[:13], energie2[:13], 'ro', label='Messdaten (Regression')
plt.plot(x2[12:], energie2[12:], 'ko', label='Messdaten')
plt.plot(x, linear(x, *parameterse2),'k-', label = 'Regression an den linearen Teil')
plt.xlim(0,2.2)
plt.legend(loc='best')
plt.xlabel('Effektiver Abstand zwischen Detektor und Strahler x in cm')
plt.ylabel('Energie in MeV')
plt.savefig('build/energie2.png')
plt.show()

#Energien berechnen
E1_reg = - m_e1 * reichweite1
E2_reg = - m_e2 * reichweite2

write('build/E1_reg.txt', make_SI(E1_reg, r'\mega\electronvolt', figures=1))
write('build/E2_reg.txt', make_SI(E2_reg, r'\mega\electronvolt', figures=1))
write('build/m_e1.txt', make_SI(m_e1, r'\mega\electronvolt\per\centi\meter', figures=1))
write('build/m_e2.txt', make_SI(m_e2, r'\mega\electronvolt\per\centi\meter', figures=1))






################Statistik des radioaktiven Zerfalls

messung3 = np.genfromtxt('Messung3.txt', unpack=True)

#Mittelwert und Standartabweichung
messung3_mittel = np.mean(messung3)
messung3_std = np.std(messung3, ddof=1)

print('Mittelwert', messung3_mittel, 'Standtartabweichung', messung3_std)

#Histogramm nicht normiert ausgeben:

plt.hist(messung3, bins=8, normed=False, facecolor='green', alpha=0.75)
plt.xlabel('Anzahl der Pulse')
plt.ylabel('Anzahl der Messungen in einem Bin')
plt.savefig('build/histogramm_unnormiert.png')
plt.show()



n, bins, patches = plt.hist(messung3, bins=8, normed=True, facecolor='green', alpha=0.75)

#datensatz für gauß (nimmt die mitte der x-achsen-intervalle):
xg = np.ones(len(bins)-1)
for i in range(0,len(bins)-1):
    xg[i] = (bins[i]+bins[i+1])/2

xp = np.round(xg)

# regressionen (gauß und poisson)



def poisson(k, z):
#    return z**k/factorial(k)*np.exp(-z)
    return scipy.stats.poisson.pmf(k, z)

def gauss(xp, s, mu):
#    return 1/(s * np.sqrt(2*np.pi))*np.exp(-1/2*((xp-mu)/s)**2)
    return scipy.stats.norm.pdf(xp, mu, s)

#hier ist der startvektor geraten
parameters_poisson, popt_poisson = curve_fit(poisson, xp, n, p0 = [np.mean(messung3)])


#der startvektor müsste den optimalen daten entsprechen
parameters_gauss, popt_gauss = curve_fit(gauss, xg, n, p0 = [np.mean(messung3), np.std(messung3)])


# auf ganzzahlige werte achten wegen der poissonverteilung!
x = np.linspace(3400, 4100, 101)

plt.plot(x, poisson(x, *parameters_poisson), 'b-', label='Poissonverteilung')
plt.plot(x, gauss(x, *parameters_gauss), 'r-', label='Gaußverteilung')

#werte plotten, die für die regression verwendet werden
plt.plot(xg, n, 'bo', label='Werte für den Fit')
plt.xlabel('Anzahl der Pulse')
plt.ylabel('Wahrscheinlichkeit')
plt.legend(loc='best')
plt.savefig('build/histogramm.png')
plt.show()

mu = ufloat(parameters_gauss[1], np.sqrt(popt_gauss[1,1]))
sigma = ufloat(parameters_gauss[0], np.sqrt(popt_gauss[0,0]))
lam = ufloat(parameters_poisson[0], np.sqrt(popt_poisson[0,0]))


write('build/mu.txt', make_SI(mu, r'\mega\electronvolt', figures=1))
write('build/sigma.txt', make_SI(sigma, r'\mega\electronvolt', figures=1))
write('build/lambda.txt', make_SI(lam, r'', figures=1))

#Werte die gefittet wurden in Tabellen schreiben:
write('build/tabelle_histogramm.txt', make_table([xg, xp, n], [2,0,4]))






