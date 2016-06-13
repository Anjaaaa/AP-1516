import numpy as np
from uncertainties import ufloat
from uncertainties import unumpy as unp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
from table import(
        make_table,
        make_SI,
        write,
        make_full_table,
        make_composed_table)

##############################################################################
### Teil A kopiert für Diskussion
##############################################################################
e1, e2, t1, t2 = np.genfromtxt('WerteA.txt', unpack=True)
print(e1)

 # in SI-Einheiten umrechnen
e1 = e1/100
e2=e2/100
t1=t1*10**-6
t2=t2*10**-6

#schallgeschwindigkeit in acryl
c = 2730

s1 = c*t1/2
s2= c*t2/2

print(np.mean(s1-e1))
print(np.mean(s2-e2))

print(s1)
print(s2[::-1])

write('build/tabelle_WerteA.txt', make_table([s1*1000, (s1-np.mean(s1-e1))*1000, s2*1000, (s2-np.mean(s2-e2))*1000], [2,2,2,2]))
write('build/WerteA.tex', make_full_table(
    r'Aus den A-Scans bestimmte Abstände der Löcher zum Rand in \si{\milli\meter}',
    'tab:werteA',
    'build/tabelle_WerteA.txt',
    [],
    [r'$d_\text{berechnet}$',
    r'$d_\text{korrigiert}$',
    r'$d_\text{berechnet}$',
    r'$d_\text{korrigiert}$']))


###############################################################################
### B-Scan
###############################################################################

tObenG, tUntenG = np.genfromtxt('WerteB.txt', unpack = True)
sOben, sUnten, tO,tU = np.genfromtxt('WerteA.txt', unpack = True)
sOben *= 0.01
sUnten *= 0.01
# 100 * 10**(-6) sekunden = 543.0 pixel
tOben = tObenG / 543 * 100 * 10**(-6)
tUnten = tUntenG / 543 * 100 * 10**(-6)

write('build/WerteZeit.tex', make_table([tObenG, tOben*10**6, tUntenG, tUnten*10**6],[0,2,0,2]))
write('build/Zeit.tex', make_full_table(
    r'Aus den B-Scans bestimmte Laufzeiten in Pixeln $\tilde{t}$ und umgerechnete Laufzeiten $t$',
    'tab:Zeit',
    'build/WerteZeit.tex',
    [],
    [r'$\tilde{t}_\text{oben}$ in \si{Pixel}',
    r'$t_\text{oben}$ in \si{\micro\second}',
    r'$\tilde{t}_\text{unten}$ in \si{Pixel}',
    r'$t_\text{unten}$ in \si{\micro\second}']))



h = 8.020 * 10**(-2)    # Höhe des Quaders
c = 2730

xOben = 0.5 * tOben * c
xUnten = 0.5 * tUnten * c


d = h - xOben - xUnten
dGemessen = h - sOben - sUnten

write('build/WerteBScan.tex', make_table([xOben*10**3, xUnten*10**3, d*10**3, dGemessen*10**3],[2,2,2,2]))
write('build/BScan.tex', make_full_table(
    r'Berechnete Abstände der Löcher zum oberen Rand $s_\text{oben}$ und zum unteren Rand $s_\text{unten}$, die daraus bestimmte Dicke der Löcher $d$ und die gemessene Dicke der Löcher. Alle Werte in \si{\milli\meter}.',
    'tab:ObenGanz',
    'build/WerteBScan.tex',
    [],
    [r'$s_\text{oben}$',
    r'$s_\text{unten}$',
    r'$d$',
    r'$\tilde{d}$']))


write('build/Vergleich.txt', make_table([(1-(s1-np.mean(s1-e1))/e1)*100, (1-xOben/e1)*100, (1-(s2-np.mean(s2-e2))/e2)*100, (1-xUnten/e2)*100], [1,1,1,1]))
write('build/Diskussion.tex', make_full_table(
    r'Abweichung $a$ der berechneten Abstände der Löcher zum oberen Rand $a_\text{oben}$ und zum unteren Rand $a_\text{unten}$ in Prozent',
    'tab:Disussion',
    'build/Vergleich.txt',
    [],
    [r'$a_\text{A,oben}$',
    r'$a_\text{B,oben}$',
    r'$a_\text{A,unten}$',
    r'$a_\text{B,unten}$']))

###############################################################################
### Herz-Modell
###############################################################################
tMaxG, tMinG = np.genfromtxt('WerteHerz.txt', unpack = True)
# 100 * 10**(-6) sekunden = 543.0 pixel
tMax = tMaxG / 543 * 100 * 10**(-6)
tMin = tMinG / 543 * 100 * 10**(-6)
dHerz = 49.45 * 10**(-3)
cWasser = 1484
t0Herz = 58.3*10**(-6)
hoeheHerz = cWasser * 0.5 * t0Herz

nu = 6/16
write('build/Hoehe.tex', make_SI(hoeheHerz*10**3, r'\milli\meter', figures=1))



write('build/WerteTMScan.tex', make_table([tMaxG, tMax*10**6, tMinG, tMin*10**6],[0,2,0,2]))
write('build/TMScan.tex', make_full_table(
    r'Aus dem TM-Scan bestimmte Laufzeiten in Pixeln $\tilde{t}$ und umgerechnete Laufzeiten $t$',
    'tab:ZeitHerz',
    'build/WerteTMScan.tex',
    [],
    [r'$\tilde{t}_{max}$ in \si{Pixel}',
    r'$t_{max}$ in \si{\micro\second}',
    r'$\tilde{t}_{min}$ in \si{Pixel}',
    r'$t_{min}$ in \si{\micro\second}']))


Max = ufloat(np.mean(tMax), np.std(tMax)/(len(tMax)-1))
Min = ufloat(np.mean(tMin), np.std(tMin)/(len(tMin)-1))

T = Min - Max

a = cWasser * 0.5 * T
write('build/Kugel.tex', make_SI(a*10**3, r'\milli\meter', figures=1))

HZV = a * np.pi / 3 * (3/4 * dHerz**2 + a**2) * nu

write('build/HZV.tex', make_SI(HZV*10**6, r'\centi\meter\cubed', figures=1))

