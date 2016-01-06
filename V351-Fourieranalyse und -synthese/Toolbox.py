import numpy as np
import matplotlib.pyplot as plt



n_S, Saegezahn = np.genfromtxt('Saegezahn.txt', unpack = True)


#Theoretische Fourierkoeffizienten a
def a_S(u_S, l):
    return - u_S / (np.pi * l) #u_D setzt sich eigentlich zusammen aus u_D*T


#FOURIERANALYSE
#die Konestanten berechnen sich aus dem ersten gemessenen Koeffizienten
u_S = - Saegezahn[0] * np.pi


# Quadratische Abweichungen berechnen
Abweichung_S = 0      
for i in range(0, len(n_S)):
    Abweichung_S = Abweichung_S + (Saegezahn[i] - a_S(u_S, n_S[i]))**2
Abweichung_S = np.sqrt(Abweichung_S)/(len(n_S)-1)
print('Die Abweichung bei der Sägezahnspannung pro Wert:',(Abweichung_S)) 



ax=plt.subplot()  #ist gut für die Achsenbeschriftung 

plt.bar(left, 10, width = 0.1, label = 'Gemessene Werte')
plt.bar(n_S-0.2, a_S(u_S, n_S), width = 0.1, color='r', label = 'Errechnete Werte')
#plt.bar(n_S-0.2, a_S(u_S, n_S), left, width = 0.1, color='r', label = 'Errechnete Werte')
plt.errorbar(n_S-0.15, a_S(u_S,n_S), yerr=Abweichung_S, color='b', fmt='none', label='Fehler auf Rechnung')
ax.set_xticks(n_S)
ax.set_xticklabels(n_S)
plt.legend(loc='best')
plt.show()


    
    
    
    
    
    
