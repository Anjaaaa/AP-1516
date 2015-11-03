import numpy as np
from uncertainties import ufloat

höhe_eckig, länge_eckig, gewicht_eckig = np.genfromtxt('eckiger_Stab.txt', unpack=True)

höhe_rund, länge_rund, gewicht_rund = np.genfromtxt('runder_Stab.txt', unpack=True)

gewicht_eckig = gewicht_eckig[np.invert(np.isnan(gewicht_eckig))] /1000
gewicht_rund = gewicht_rund[np.invert(np.isnan(gewicht_rund))] /1000

länge_eckig = länge_eckig[np.invert(np.isnan(länge_eckig))] /1000
länge_rund = länge_rund[np.invert(np.isnan(länge_rund))] /1000

#ablesefehler ist 0.05
höhe_eckig_gesamt = ufloat(np.mean(höhe_eckig), np.std(höhe_eckig)/np.sqrt(len(höhe_eckig))+ 0.05) /1000
höhe_rund_gesamt = ufloat(np.mean(höhe_rund), np.std(höhe_rund)/np.sqrt(len(höhe_rund))+ 0.05) /1000


#flächenträgheitsmomente bestimmen
I_eckig = (höhe_eckig_gesamt)**4 / 12
I_rund = (höhe_rund_gesamt / 2)**4 *np.pi / 4

print ('Flächenträgheitsmoment des eckigen Stabes:', I_eckig)
print ('Flächenträgheitsmement des runden Stabes:', I_rund, '\n')

#dichten ausrechen
rho_eckig = gewicht_eckig / ((höhe_eckig_gesamt)**2 * länge_eckig)
rho_rund = gewicht_rund / ( np.pi * (höhe_rund_gesamt / 2)**2 * länge_rund)

print('Dichte des eckigen Stabes:', rho_eckig)
print ('Dichte des runden Stabes:', rho_rund)
