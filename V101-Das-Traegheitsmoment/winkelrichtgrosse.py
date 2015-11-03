import numpy as np

phi, F = np.genfromtxt('winkelrichtgrosse_Winkel und Kraft.txt', unpack='True')
r = 0.2884 #m

phi_bog = phi/360*2*np.pi
D = F*r/phi_bog

D_mittel= np.sum(D)/len(F)
D_mittel2=sum(D)/10

print (D, D_mittel)
print(np.std(D)/np.sqrt(len(D)))
