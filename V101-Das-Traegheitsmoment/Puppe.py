import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp

arm_dicke, bein_dicke, rumpf_dicke, kopf_dicke, arm_lange, bein_lange, rumpf_lange, kopf_lange, gewicht, periode = np.genfromtxt('Puppe.txt', unpack=True)

arm_dicke = arm_dicke[np.invert(np.isnan(arm_dicke))] /1000
bein_dicke = bein_dicke[np.invert(np.isnan(bein_dicke))] /1000
rumpf_dicke = rumpf_dicke[np.invert(np.isnan(rumpf_dicke))] /1000
kopf_dicke = kopf_dicke[np.invert(np.isnan(kopf_dicke))] /1000
arm_lange = arm_lange[np.invert(np.isnan(arm_lange))] /1000
bein_lange = bein_lange[np.invert(np.isnan(bein_lange))] /1000
rumpf_lange = rumpf_lange[np.invert(np.isnan(rumpf_lange))] /1000
kopf_lange = kopf_lange[np.invert(np.isnan(kopf_lange))] /1000
gewicht = gewicht[np.invert(np.isnan(gewicht))] /1000
periode = periode[np.invert(np.isnan(periode))]


#Varianz = 0
#Fehler zu Fuß ausrechnen
#for i in range(0, len(arm_dicke)):
#    a = (arm_dicke[i] - arm_dicke_average)**2
#    Varianz = Varianz + a
#    print (Varianz, '\n')
#    i = i+1


#Varianz = Varianz/(len(arm_dicke)) #-1 wegen Messung bei Stichprobe
#Standartabweichung = np.sqrt(Varianz)

#print(np.average(arm_dicke), Varianz, Standartabweichung, '\n')



arm_dicke_gesamt = ufloat(np.mean(arm_dicke), np.std(arm_dicke))
arm_lange_gesamt = ufloat(np.mean(arm_lange), np.std(arm_lange))

bein_dicke_gesamt = ufloat(np.mean(bein_dicke), np.std(bein_dicke))
bein_lange_gesamt = ufloat(np.mean(bein_lange), np.std(bein_lange))

rumpf_dicke_gesamt = ufloat(np.mean(rumpf_dicke), np.std(rumpf_dicke))
rumpf_lange_gesamt = ufloat(np.mean(rumpf_lange), np.std(rumpf_lange))

kopf_dicke_gesamt = ufloat(np.mean(kopf_dicke), np.std(kopf_dicke))
kopf_lange_gesamt = ufloat(np.mean(kopf_lange), np.std(kopf_lange))

print ('Arm Dicke: {:.5u}' .format(arm_dicke_gesamt))
print ('Arm Länge:', arm_lange_gesamt)
print ('Bein Dicke:', bein_dicke_gesamt)
print ('Bein Länge:', bein_lange_gesamt)
print ('Rumpf Dicke:', rumpf_dicke_gesamt)
print ('Rumpf Länge:', rumpf_lange_gesamt, '\n')
print ('Kopf Dicke:', kopf_dicke_gesamt)
print ('Kopf Länge:', kopf_lange_gesamt)
#Volumina bestimmen:

volumen_arm = np.pi * arm_dicke_gesamt * arm_lange_gesamt
volumen_bein = np.pi * bein_dicke_gesamt * bein_lange_gesamt
volumen_rumpf = np.pi * rumpf_dicke_gesamt * rumpf_lange_gesamt
volumen_kopf = np.pi * kopf_dicke_gesamt * kopf_lange_gesamt

print ('Volumen Arm:', volumen_arm)
print ('Volumen Bein:', volumen_bein)
print ('Volumen Rumpf:', volumen_rumpf)
print ('Volumen Kopf:', volumen_kopf, '\n')


#Massenanteile bestimmen
volumen_gesamt = 2 * volumen_arm + 2 * volumen_bein + volumen_rumpf + volumen_kopf

masse_arm = gewicht * volumen_arm / volumen_gesamt
masse_bein = gewicht * volumen_bein / volumen_gesamt
masse_rumpf = gewicht * volumen_rumpf / volumen_gesamt
masse_kopf = gewicht * volumen_kopf / volumen_gesamt

masse_gesamt = 2 * masse_arm + 2 * masse_bein + masse_rumpf + masse_kopf

print ('Gewicht ein (!) Arm:', masse_arm)
print ('Gewicht ein (!) Bein:', masse_bein)
print ('Gewicht Rumpf:', masse_rumpf)
print ('Gewicht Kopf:', masse_kopf, '\n')
print (masse_gesamt, gewicht, '\n')

#Trägheitsmoment bestimmen (Puppe hat rechtwinklig abgespreizte Arme und Beine)
I_rumpf = masse_rumpf * (rumpf_dicke_gesamt / 2)**2 /2
I_kopf = masse_kopf * (kopf_dicke_gesamt / 2)**2 /2
I_bein = masse_bein * (bein_dicke_gesamt / 2) **2 / 4 + (bein_lange_gesamt)**2 / 12
I_arm = masse_bein * (arm_dicke_gesamt / 2)**2 / 4 + (arm_lange_gesamt)**2 / 12

print ('Trägheitsmomente:', I_rumpf, I_kopf, I_bein, I_arm, '\n')

I_gesamt = I_rumpf + I_kopf \
+ 2 * (I_bein + masse_bein * (bein_lange_gesamt / 2)**2) \
+ 2 * (I_arm * masse_arm * ((arm_lange_gesamt + rumpf_dicke_gesamt) / 2)**2) #Satz von Steiner

print ('Gesamtträgheitsmoment Theorie:', I_gesamt)

periode_puppe = ufloat(np.mean(periode), np.std(periode))
D = 0.02857 #aus winkelrichtgrose.py

print (periode_puppe)
I_experiment = (periode_puppe / 2 / np.pi)**2 * D
print ('Trägheitsmoment Experiment:', I_experiment)
print ('experiment-theorie:', I_experiment - I_gesamt)
