import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import unumpy as unp
from uncertainties.umath import *
from table import(
        make_table,
        make_SI,
        make_full_table,
        write)


NullWinkel, wavelength, PhiHe, PhiNa, PhiKa, PhiRu, sHe, sNa, sKa, sRu = np.genfromtxt('Werte.txt', unpack = True)

##########################################################################################################
### Messwerte auf geeignete Form bringen #################################################################

wavelength = wavelength * 10**(-9)
NullWinkel = NullWinkel[np.invert(np.isnan(NullWinkel))]

# Diese Rechnungen sind aus dem Protokoll von Marius und Matthias. Ich habe keine Ahnung warum das so geht.
Beta = 90-0.5*(400-NullWinkel)
PhiHe = NullWinkel - Beta - PhiHe
PhiNa = NullWinkel - Beta - PhiNa
PhiKa = NullWinkel - Beta - PhiKa
PhiRu = NullWinkel - Beta - PhiRu


write('build/Winkel.tex', make_table([PhiNa, PhiKa, PhiRu],[1,1,1]))
write('build/WinkelGanz.tex', make_full_table(
    r'Gemessene Beugungswinkel $\varphi$ in \si{\degree} von Natrium, Kalium und Rubidium',
    'tab:Winkel',
    'build/Winkel.tex',
    [],
    [r'$\varphi_\text{\ce{Na}}$',
    r'$\varphi_\text{\ce{K}}$',
    r'$\varphi_\text{\ce{Ru}}$']))


PhiNa = PhiNa[np.invert(np.isnan(PhiNa))]
PhiKa = PhiKa[np.invert(np.isnan(PhiKa))]
PhiRu = PhiRu[np.invert(np.isnan(PhiRu))]
sHe = sHe[np.invert(np.isnan(sHe))]
sNa = sNa[np.invert(np.isnan(sNa))]
sKa = sKa[np.invert(np.isnan(sKa))]
sRu = sRu[np.invert(np.isnan(sRu))]


# Umrechnen in Bogenmaß
PhiHeDeg = PhiHe # Phi möchte ich in der Tabelle in Grad haben
PhiHe = 2*np.pi/360 * PhiHe
PhiNa = 2*np.pi/360 * PhiNa
PhiKa = 2*np.pi/360 * PhiKa
PhiRu = 2*np.pi/360 * PhiRu


##########################################################################################################
### Bestimmung der Gitterkonstante durch Regression ######################################################

def f(X, m, t):
    return m * X + t

Y = wavelength
X = np.sin(PhiHe)

params, cov = curve_fit(f, X, Y)
Steigung = ufloat(params[0], np.sqrt(cov[0,0]))
Verschiebung = ufloat(params[1], np.sqrt(cov[1,1]))
m = params[0]
t = params[1]


plt.plot(X, Y*10**9, 'ro', label = 'Datenpunkte')
plt.plot(X, f(X, m, t)*10**9, 'k', label = 'Regressionsgerade')
plt.xlabel(r'$\sin(\varphi)$')
plt.ylabel(r'Wellenlänge $\lambda / \mathrm{nm}$')
plt.legend(loc='best')
plt.xlim(-0.4,0)

plt.savefig('Regression.png')
#plt.show()


write('build/Regression.tex', make_SI(Verschiebung*10**9, r'\nano\meter', figures=1))
write('build/Gitterkonstante.tex', make_SI(Steigung*10**9, r'\nano\meter', figures=1))

write('build/Helium.tex', make_table([wavelength*10**9, PhiHeDeg, X],[1,1,3]))
write('build/HeliumGanz.tex', make_full_table(
    r'Gemessene Beugungswinkel je Wellenlänge und Werte $\sin(\varphi)$ für die Regression',
    'tab:Helium',
    'build/Helium.tex',
    [],
    [r'$\lambda \ \mathrm{in} \ \si{\nano\meter}$',
    r'$\varphi_\text{\ce{He}} \ \mathrm{in} \ \si{\degree}$',
    r'$\sin(\varphi)$']))



##########################################################################################################
### Eichgröße ############################################################################################

lambda1 = wavelength[6]       # 6-te Spektrallinie laut Messheft
lambda2 = wavelength[8]       # 8-te Spektrallinie laut Messheft

phiMittel = ufloat(np.mean([PhiHe[6], PhiHe[8]]), np.std([PhiHe[6], PhiHe[8]])/np.sqrt(2))

Xi = (lambda2-lambda1) / sHe / unp.cos(phiMittel)
#write('build/Eichgrosse.tex', make_SI(Eichgrosse*10**9, r'\nano\meter', figures=1))
print(Xi)


##########################################################################################################
### Abschirmungszahlen ###################################################################################

# Delta Lambda berechnen
def D_wavelength(Phi, s):
   return Xi*unp.cos(Phi)*s

D_wavelengthNa = D_wavelength(PhiNa, sNa)
D_wavelengthKa = D_wavelength(PhiKa, sKa)
D_wavelengthRu = D_wavelength(PhiRu, sRu)


# Lambda berechnen
def wavelength(Phi):
   return unp.sin(Phi) * Steigung + Verschiebung

wavelengthNa = wavelength(PhiNa)
wavelengthKa = wavelength(PhiKa)
wavelengthRu = wavelength(PhiRu)


# Delta E berechnen
c = 299792458
h = 6.626070040*10**(-34)
def E(D_wavelength, wavelength):
   return h*c*D_wavelength/wavelength**2

ENa = E(D_wavelengthNa, wavelengthNa)
EKa = E(D_wavelengthKa, wavelengthKa)
ERu = E(D_wavelengthRu, wavelengthRu)


# Abschirmungszahl berechnen
e0 = 1.6021766208*10**(-19)
epsilon0 = 8.854187817*10**(-12)
a = e0**2 / 2 / h / c / epsilon0     # Sommerfeldsche Feinstrukturkonstante
R = 10973731.568508
def sigma(z, E, n):
   Wurzel = E * 2 * n**3 / R / a**2
   return z - Wurzel**(1/4)

sigmaNa = sigma(11, ENa, 3)
sigmaKa = sigma(19, EKa, 4)
sigmaRu = sigma(37, ERu, 5)

write('build/AbschirmungszahlNatrium.tex', make_table([wavelengthNa*10**9, D_wavelengthNa*10**9, sNa, ENa/e0*10**3, sigmaNa],[1,1,0,1,1]))
write('build/AbschirmungszahlNa.tex', make_full_table(
    r'Natrium -- Abschirmungszahl für jedes betrachtete Duplett, sowie bei der Berechnung verwendeten Größen',
    'tab:Natrium',
    'build/AbschirmungszahlNatrium.tex',
    [0, 1, 3, 4],
    [r'$\lambda \ \mathrm{in} \ \si{\nano\meter}$',
    r'$\Delta\lambda \ \mathrm{in} \ \si{\nano\meter}$',
    r'$\Delta s \ \mathrm{in} \ \mathrm{Skt}$',
    r'$\Delta E_\text{D} \ \mathrm{ in } \ \si{\milli\electronvolt}$',
    r'$\sigma_2$']))

write('build/AbschirmungszahlKalium.tex', make_table([wavelengthKa*10**9, D_wavelengthKa*10**9, sKa, EKa/e0*10**3, sigmaKa],[1,1,0,1,1]))
write('build/AbschirmungszahlKa.tex', make_full_table(
    r'Kalium -- Abschirmungszahl für jedes betrachtete Duplett, sowie bei der Berechnung verwendeten Größen',
    'tab:Kalium',
    'build/AbschirmungszahlKalium.tex',
    [0, 1, 3, 4],
    [r'$\lambda \ \mathrm{ in } \ \si{\nano\meter}$',
    r'$\Delta\lambda \ \mathrm{ in } \ \si{\nano\meter}$',
    r'$\Delta s \ \mathrm{in} \ \mathrm{Skt}$',
    r'$\Delta E_\text{D} \ \mathrm{ in } \ \si{\milli\electronvolt}$',
    r'$\sigma_2$']))
write('build/AbschirmungszahlRubidium.tex', make_table([wavelengthRu*10**9, D_wavelengthRu*10**9, sRu, ERu/e0*10**3, sigmaRu],[1,1,0,1,1]))

write('build/AbschirmungszahlRu.tex', make_full_table(
    r'Rubidium -- Abschirmungszahl für das betrachtete Duplett, sowie bei der Berechnung verwendeten Größen',
    'tab:Rubidium',
    'build/AbschirmungszahlRubidium.tex',
    [0, 1,  3, 4],
    [r'$\lambda \ \mathrm{ in } \ \si{\nano\meter}$',
    r'$\Delta\lambda \ \mathrm{ in } \ \si{\nano\meter}$',
    r'$\Delta s \ \mathrm{in} \ \mathrm{Skt}$',
    r'$\Delta E_\text{D} \ \mathrm{ in } \ \si{\milli\electronvolt}$',
    r'$\sigma_2$']))



sigmaNaMittel = np.mean(sigmaNa)
sigmaKaMittel = np.mean(sigmaKa)
sigmaRuMittel = np.mean(sigmaRu)




write('build/AbschirmungNaMittel.tex', make_SI(sigmaNaMittel, r'', figures=1))
write('build/AbschirmungKaMittel.tex', make_SI(sigmaKaMittel, r'', figures=1))
write('build/AbschirmungRuMittel.tex', make_SI(sigmaRuMittel, r'', figures=1))


