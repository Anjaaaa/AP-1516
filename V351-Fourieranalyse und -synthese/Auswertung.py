import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


#Gemessene Fourierkoeffizienten einlesen
n_D, Dreieck = np.genfromtxt('Dreieck.txt', unpack=True)
n_R, Rechteck = np.genfromtxt('Rechteck.txt', unpack = True)
n_S, Saegezahn = np.genfromtxt('Saegezahn.txt', unpack = True)


#Theoretische Fourierkoeffizienten a
def a_D(u_D,n):
    return 8 * u_D / (np.pi * n)**2
def a_R(u_R, m):
    return 4 * u_R / (np.pi * m)
def a_S(u_S, l):
    return - u_S / (np.pi * l) #u_D setzt sich eigentlich zusammen aus u_D*T


#FOURIERANALYSE
#die Konestanten berechnen sich aus dem ersten gemessenen Koeffizienten
u_D = Dreieck[0] * np.pi**2 / 8
u_R = Rechteck[0] * np.pi / 4
u_S = - Saegezahn[0] * np.pi


# Quadratische Abweichungen berechnen
Abweichung_D = 0
Abweichung_R = 0
Abweichung_S = 0

for i in range(1, len(n_D)):
    Abweichung_D = Abweichung_D + (Dreieck[i] - a_D(u_D, n_D[i]))**2
Abweichung_D = np.sqrt(Abweichung_D)/(len(n_D)-2)
print('Die Abweichung bei der Dreieckspannung pro Wert:', Abweichung_D)    

    
for i in range(1, len(n_R)):
    Abweichung_R = Abweichung_R + (Rechteck[i] - a_R(u_R, n_R[i]))**2
Abweichung_R =  np.sqrt(Abweichung_R)/(len(n_R)-2)   
print('Die Abweichung bei der Rechteckspannung pro Wert:', Abweichung_R)    
    
for i in range(1, len(n_S)):
    Abweichung_S = Abweichung_S + (Saegezahn[i] - a_S(u_S, n_S[i]))**2
Abweichung_S = np.sqrt(Abweichung_S)/(len(n_S)-2)
print('Die Abweichung bei der Sägezahnspannung pro Wert:',(Abweichung_S)) 



ax1=plt.subplot()  #ist gut für die Achsenbeschriftung 


plt.bar(n_D, Dreieck, width = 0.1, label = 'Gemessene Werte')
plt.bar(n_D-0.1, a_D(u_D, n_D), width = 0.1, color='r', label = 'Errechnete Werte')
plt.errorbar(n_D-0.05, a_D(u_D,n_D), yerr=Abweichung_D, color='b', fmt='none', label='Mittlere Abweichung')
ax1.set_xticks(n_D)
ax1.set_xticklabels(n_D)
plt.legend(loc='best')
plt.savefig('Dreieck_Fourier.pdf')
plt.show()

ax2=plt.subplot()
plt.bar(n_R, Rechteck, width = 0.4, label = 'Gemessene Werte')
plt.bar(n_R-0.4, a_R(u_R, n_R), width = 0.4, color='r', label = 'Errechnete Werte')
plt.errorbar(n_R-0.2, a_R(u_R,n_R), yerr=Abweichung_R, color='b', fmt='none', label='Mittlere Abweichung')
ax2.set_xticks(n_R)
ax2.set_xticklabels(n_R)
plt.legend(loc='best')
plt.savefig('Rechteck_Fourier.pdf')
plt.show()

ax3=plt.subplot()
plt.bar(n_S, Saegezahn, width = 0.3, label = 'Gemessene Werte')
plt.bar(n_S-0.3, a_S(u_S, n_S), width = 0.3, color='r', label = 'Errechnete Werte')
plt.errorbar(n_S-0.15, a_S(u_S,n_S), yerr=Abweichung_S, color='b', fmt='none', label='Mittlere Abweichung')
ax3.set_xticks(n_S)
ax3.set_xticklabels(n_S)
plt.legend(loc='best')
plt.savefig('Saegezahn_Fourier.pdf')
plt.show()


#FOURIERSYNTHESE

#Koeffizienten die wir eingestellt haben, hatten die Startwerte in mV
u_D2 = 775.8
u_R2 = 495.6
u_S2 = 1970.1

x = np.arange(1,10)
Synthese_D = a_D(u_D2, x)
Synthese_R = a_R(u_R2, x)
Synthese_S = a_S(u_S2, x)

Koeffizienten = np.array(([Synthese_D], [Synthese_R], [Synthese_S])

#f = open('Synthese_Koeffizienten.txt', 'w')
#f.write(tabulate(table.T, tablefmt="latex"))


    
    
    
    
    
    
