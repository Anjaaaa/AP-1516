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


OffSetBrom = 182/900
OffSetSilber = 222/1200

Silber, Brom = np.genfromtxt('Messwerte.txt', unpack = True)

Brom = Brom[np.invert(np.isnan(Brom))]

Brom = Brom - OffSetBrom*180
Silber = Silber - OffSetSilber*10

BromAnf = Brom
SilberAnf = Silber

def Regression(x, m, b):
   return m*x + b

###############################################################################################################
### Brom ######################################################################################################
###############################################################################################################

tBrom = np.array(range(1,11))
tB = np.linspace(0, 33, num=10)

paramsBrom, poptBrom = curve_fit(Regression, tBrom*3, np.log(Brom), sigma = np.log(np.sqrt(Brom)))

TBrom = -np.log(2)/ufloat(paramsBrom[0], poptBrom[0,0])
write('build/BromT.tex', make_SI(TBrom, r'\second', figures=1))

plt.errorbar(tBrom*3, Brom, xerr=0, yerr=np.sqrt(Brom), fmt='ko', label = 'Messdaten')
plt.plot(tB, np.exp(Regression(tB, paramsBrom[0], paramsBrom[1])), 'r', label = 'Regressionsgerade')

plt.xticks(np.arange(0, 3*max(tBrom)+3, 3.0))
plt.legend(loc='best')
plt.xlabel('Zeit $t$ in Minuten')
plt.ylabel('Anzahl der Pulse in $180s$ ')
plt.yscale('log')
plt.savefig('build/Brom.png')
plt.show()


###############################################################################################################
### Silber ####################################################################################################
###############################################################################################################

tSilber = np.array(range(1,43))

### Entfernen von blöden Werten ###############################################################################
SilberOriginal = Silber
Silber = np.delete(Silber, 39)
Silber = np.delete(Silber, 37)
Silber = np.delete(Silber, 35)
Silber = np.delete(Silber, 33)
Silber = np.delete(Silber, 28)
Silber = np.delete(Silber, 27)
tSilber = np.delete(tSilber, 39)
tSilber = np.delete(tSilber, 37)
tSilber = np.delete(tSilber, 35)
tSilber = np.delete(tSilber, 33)
tSilber = np.delete(tSilber, 28)
tSilber = np.delete(tSilber, 27)


SilberLang = Silber
tSilberLang = tSilber
SilberKurz = Silber
tSilberKurz = tSilber


### Teilen in langer Zerfall und kurzer Zerfall ###############################################################
SilberLang = np.delete(SilberLang, range(0,12))
tSilberLang = np.delete(tSilberLang, range(0,12))
SilberKurz = np.delete(SilberKurz, range(9,39))
tSilberKurz = np.delete(tSilberKurz, range(9,39))


### Ignorierte Werte mitnehmen ################################################################################
Ignoriert = np.array([SilberOriginal[9], SilberOriginal[10], SilberOriginal[11], SilberOriginal[27], SilberOriginal[28], SilberOriginal[33], SilberOriginal[35], SilberOriginal[37], SilberOriginal[39]])
tIgnoriert = np.array([10,11,12,28,29,34,36,38,40]) # muss immer eins größer sein, als der Index bei Ignoriert, weil der Index bei 0 los geht



### Regression langer Zerfall
paramSL, poptSL = curve_fit(Regression, tSilberLang*10, np.log(SilberLang), sigma = np.log(np.sqrt(SilberLang)))

### Regression kurzer Zerfall
SilberKurzReg = SilberKurz - Regression(tSilberKurz, paramSL[0], paramSL[1])     # Abziehen des Langen Zerfalls
paramSK, poptSK = curve_fit(Regression, tSilberKurz*10, np.log(SilberKurzReg), sigma = np.log(np.sqrt(SilberKurzReg)))

TLang = -np.log(2)/ufloat(paramSL[0], poptSL[0,0])
write('build/SilberLangT.tex', make_SI(TLang, r'\second', figures=1))
TKurz = -np.log(2)/ufloat(paramSK[0], poptSK[0,0])
write('build/SilberKurzT.tex', make_SI(TKurz, r'\second', figures=1))



### Plotten ###################################################################################################
tSL = np.linspace(130, 430, num=20)   # Intervall auf dem der lange Zerfall geplottet wird
tSK = np.linspace(0, 90, num = 20)    # Intervall auf dem der kurze Zerfall geplottet wird
t = np.linspace(0, 430, num=20)

plt.errorbar(tSilberLang*10, SilberLang, xerr=0, yerr=np.sqrt(SilberLang), fmt='ro', label = 'Langer Zerfall')
plt.errorbar(tSilberKurz*10, SilberKurzReg, xerr=0, yerr=np.sqrt(SilberKurzReg), fmt='bo', label = 'Kurzer Zerfall')
plt.plot(tSilberKurz*10, SilberKurz, 'k.', label = 'unkorrigierte Werte')
plt.plot(tIgnoriert*10, Ignoriert, 'ko', label = 'Ausgenommene Werte')


plt.plot(tSL, np.exp(Regression(tSL, paramSL[0], paramSL[1])), 'r')
plt.plot(tSK, np.exp(Regression(tSK, paramSK[0], paramSK[1])), 'b')


plt.legend(loc='best')
plt.xlabel('$t / s$')
plt.ylabel('$N / 10s$ ')
plt.yscale('log')

plt.xlim([0,430])
plt.savefig('build/Silber.png')
plt.show()


### addierte Kurven ###########################################################################################
t = np.linspace(0, 430, num=20)
tAll = np.array(range(1,43))

plt.plot(tAll*10, SilberOriginal, 'ko', label = 'Messwerte')
plt.plot(t, np.exp(Regression(t, paramSL[0], paramSL[1])) + np.exp(Regression(t, paramSK[0], paramSK[1])), 'g', label = 'Addierte Regressionsgeraden')

plt.plot(t, np.exp(Regression(t, paramSL[0], paramSL[1])), 'r')
plt.plot(t, np.exp(Regression(t, paramSK[0], paramSK[1])), 'b')

plt.legend(loc='best')
plt.xlabel('$t / s$')
plt.ylabel('$N / 10s$ ')
plt.yscale('log')

plt.xlim([0,430])
plt.savefig('build/SilberAddiert.png')
plt.show()



write('build/Brom.tex', make_SI(ufloat(paramsBrom[0],np.sqrt(poptBrom[0,0])), r'\per\second', figures=1))
write('build/SilberKurz.tex', make_SI(ufloat(paramSK[0],np.sqrt(poptSK[0,0])), r'\per\second', figures=1))
write('build/SilberLang.tex', make_SI(ufloat(paramSL[0],np.sqrt(poptSL[0,0])), r'\per\second', figures=1))
write('build/BromB.tex', make_SI(ufloat(paramsBrom[1],np.sqrt(poptBrom[1,1])), r'', figures=1))
write('build/SilberKurzB.tex', make_SI(ufloat(paramSK[1],np.sqrt(poptSK[1,1])), r'', figures=1))
write('build/SilberLangB.tex', make_SI(ufloat(paramSL[1],np.sqrt(poptSL[1,1])), r'', figures=1))



Silber = unp.uarray(np.zeros(len(SilberAnf)), np.zeros(len(SilberAnf)))
Brom = unp.uarray(np.zeros(len(BromAnf)), np.zeros(len(BromAnf)))

for i in range(0, len(SilberAnf)):
   Silber[i] = ufloat(SilberAnf[i], np.sqrt(SilberAnf[i]))
for i in range(0, len(BromAnf)):
   Brom[i] = ufloat(BromAnf[i], np.sqrt(BromAnf[i]))


write('build/Werte.tex', make_table([Silber,Brom],[1,1]))

