import numpy as np
from uncertainties import ufloat
from uncertainties.umath import *

# 1 ist immer das, was weiter weg von der Heizung war

########## Wärmestrom in Messing mit Formel 1 bei statischer Messung
x = 0.03
t = np.array([50, 100, 150, 200, 250, 300])
# 2.4 cm = 1°  ->  1cm = 1/2.4
T = np.array([4+2.15/2.35, 4+1.4/2.35, 3+2.25/2.35, 3+1.15/2.35, 3+0.45/2.35, 2+2.3/2.35])
print('Temperaturdifferenz:', T)
k = 112
A = 0.012*0.004

Strom = -k*A*(T/t)

print('Wärmestrom:', ufloat(np.mean(Strom), np.std(Strom)/len(Strom)), '\n', '\n', '\n')




######### Phasendifferenz, Amplituden und Wärmeleitfähigkeit bei Messing bei dynamischer Methode
# 2.4 cm = 100 s  ->  1 cm = 100/2.4
# 1.65 cm = 5°  ->  1 cm = 5/1.65

Delta_tM = np.array([0.2, 0.15, 0.15, 0.2, 0.2, 0.2, 0.2, 0.2])	# Abstand der Minima in cm
Delta_tM = Delta_tM*100 / 2.4
Delta_tM = ufloat(np.mean(Delta_tM), np.std(Delta_tM)/len(Delta_tM))


A2_kleinM = np.array([1.7, 1.7, 1.75, 1.95, 1.85, 1.7, 2, 2])
A2_kleinM = A2_kleinM*5 / 1.65
A2_großM = np.array([2.6, 2.8, 2.85, 2.75, 2.7, 2.35, 2.7, 2.5])
A2_großM = A2_großM*5 / 1.65
A2M = (A2_kleinM+A2_großM)/4
A2M = ufloat(np.mean(A2M), np.std(A2M)/len(A2M))


A1_kleinM = np.array([0.2, 0.2, 0.25, 0.3, 0.3, 0.3, 0.4, 0.4])
A1_kleinM = A1_kleinM*5 / 1.65
A1_großM = np.array([1.1, 1.25, 1.25, 1.05, 1.1, 0.95, 1.05, 0.9])
A1_großM = A1_großM*5 / 1.65
A1M = (A1_kleinM+A1_großM)/4
A1M = ufloat(np.mean(A1M), np.std(A1M)/len(A1M))


print('Delta t Messing:', Delta_tM)
print('Amplitude der Temperatur 1 Messing:', A1M)
print('Amplitude der Temperatur 2 Messing:', A2M)


rho_Messing = 8520
c_Messing = 385
k_Messing = (rho_Messing*c_Messing*x**2) / (2*Delta_tM*log(A2M/A1M))

print('Wärmeleitfähigkeit Messing:', k_Messing)



print('\n', '\n', '\n')

######### Phasendifferenz, Amplituden und Wärmeleitfähigkeit bei Aluminium bei dynamischer Methode
# 2.4 cm = 100 s  ->  1 cm = 100/2.4
# 2.05 cm = 5°  ->  1 cm = 5/2.05
Delta_tA = np.array([0.15, 0.15, 0.15, 0.15, 0.2, 0.2, 0.2, 0.2]) # Abstand der Minima in cm
Delta_tA = Delta_tA*100 / 2.4
print('Delta_tA:', Delta_tA)
Delta_tA = ufloat(np.mean(Delta_tA), np.std(Delta_tA)/len(Delta_tA))


A2_kleinA = np.array([1.8, 1.9, 2.0, 2.2, 2.15, 2.1, 2.35, 2.4])
A2_kleinA = A2_kleinA*5 / 2.05
A2_großA = np.array([3, 3.3, 3.3, 3, 3.2, 2.8, 3.2, 2.9])
A2_großA = A2_großA*5 / 2.05
A2A = (A2_kleinA+A2_großA)/4
print('Amplitude 2:', A2A)
A2A = ufloat(np.mean(A2A), np.std(A2A)/len(A2A))


A1_kleinA = np.array([0.65, 0.7, 0.8, 1, 0.9, 0.9, 1.1, 1.1])
A1_kleinA = A1_kleinA*5 / 2.05
A1_großA = np.array([1.95, 2.1, 2.2, 1.8, 1.95, 1.7, 1.95, 1.7])
A1_großA = A1_großA*5 / 2.05
A1A = (A1_kleinA+A1_großA)/4
print('Amplitude 1:', A1A)
A1A = ufloat(np.mean(A1A), np.std(A1A)/len(A1A))


print('Delta t Aluminium:', Delta_tA)
print('Amplitude der Temperatur 1 Aluminium:', A1A)
print('Amplitude der Temperatur 2 Aluminium:', A2A)


rho_Aluminium = 2800
c_Aluminium = 830
k_Aluminium = (rho_Aluminium*c_Aluminium*x**2) / (2*Delta_tA*log(A2A/A1A))

print('Wärmeleitfähigkeit Aluminium:', k_Aluminium)




print('\n', '\n', '\n')

######### Phasendifferenz, Amplituden und Wärmeleitfähigkeit bei Edelstahl bei dynamischer Methode
# 1 = 8 und 2 = 7
# 3.65 cm = 500 s  ->  1 cm = 500/3.65
# 2.65 cm = 10°  ->  1 cm = 10/2.65
Delta_tE = np.array([0.3, 0.25, 0.3, 0.3, 0.3, 0.35, 0.35, 0.3, 0.4, 0.35, 0.35]) # Abstand der Minima in cm
Delta_tE = Delta_tE*500 / 3.65
print('Delta_tE:', Delta_tE)
Delta_tE = ufloat(np.mean(Delta_tE), np.std(Delta_tE)/len(Delta_tE))


A1_kleinE = np.array([0, 0.05, 0.1, 0.1, 0.2, 0.25, 0.25, 0.25, 0.3, 0.3, 0.35])
A1_kleinE = A1_kleinE*5 / 2.65
A1_großE = np.array([1.1, 1, 0.85, 0.85, 0.7, 0.65, 0.6, 0.45, 0.5, 0.5, 0.5])
A1_großE = A1_großE*5 / 2.65
A1E = (A1_kleinE+A1_großE)/4
print('Amplitude 1:', A1E)
A1E = ufloat(np.mean(A1E), np.std(A1E)/len(A1E))


A2_kleinE = np.array([1.85, 2.1, 2.25, 2.95, 2.5, 2.65, 2.65, 2.7, 2.7, 2.65, 2.8])
A2_kleinE = A2_kleinE*5 / 2.65
A2_großE = np.array([3.5, 3.4, 3.2, 3.2, 3.15, 3.1, 3, 2.85, 2.8, 3, 3])
A2_großE = A2_großE*5 / 2.65
A2E = (A2_kleinE+A2_großE)/4
print('Amplitude 2:', A2E)
A2E = ufloat(np.mean(A2E), np.std(A2E)/len(A2E))


print('Delta t Edelstahl:', Delta_tE)
print('Amplitude der Temperatur 1 Edelstahl:', A1E)
print('Amplitude der Temperatur 2 Edelstahl:', A2E)


rho_Edelstahl = 8000
c_Edelstahl = 400
k_Edelstahl = (rho_Edelstahl*c_Edelstahl*x**2) / (2*Delta_tE*log(A2E/A1E))

print('Wärmeleitfähigkeit Edelstahl:', k_Edelstahl)
