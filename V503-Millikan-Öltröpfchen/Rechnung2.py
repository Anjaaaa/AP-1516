import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
import matplotlib as mpl
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties.umath import *
from uncertainties import unumpy as un
import csv
from table import(
        make_table,
        make_SI,
        write)


# 250 V: A, B, C, D
# 280 V: E, F, G, H
# 290 V: I, J, K
# 300 V: L, M, N

# t_0   t_ab       t_auf    R   U
ATime0, ATimeUp, ATimeDown = np.genfromtxt('Anton.txt', unpack = True)
BTime0, BTimeUp, BTimeDown = np.genfromtxt('Berta.txt', unpack = True)
CTime0, CTimeUp, CTimeDown = np.genfromtxt('Caesar.txt', unpack = True)
DTime0, DTimeUp, DTimeDown = np.genfromtxt('Dora.txt', unpack = True)
ETime0, ETimeUp, ETimeDown = np.genfromtxt('Emil.txt', unpack = True)
FTime0, FTimeUp, FTimeDown = np.genfromtxt('Friedrich.txt', unpack = True)
GTime0, GTimeUp, GTimeDown = np.genfromtxt('Gustav.txt', unpack = True)
HTime0, HTimeUp, HTimeDown = np.genfromtxt('Heinrich.txt', unpack = True)
ITime0, ITimeUp, ITimeDown = np.genfromtxt('Ida.txt', unpack = True)
JTime0, JTimeUp, JTimeDown = np.genfromtxt('Julius.txt', unpack = True)
KTime0, KTimeUp, KTimeDown = np.genfromtxt('Kaufmann.txt', unpack = True)
LTime0, LTimeUp, LTimeDown = np.genfromtxt('Ludwig.txt', unpack = True)
MTime0, NTimeUp, MTimeDown = np.genfromtxt('Martha.txt', unpack = True)
NTime0, MTimeUp, NTimeDown = np.genfromtxt('Nordpol.txt', unpack = True)




ATime0 = ATime0[np.invert(np.isnan(ATime0))]
BTime0 = BTime0[np.invert(np.isnan(BTime0))]
CTime0 = CTime0[np.invert(np.isnan(CTime0))]
DTime0 = DTime0[np.invert(np.isnan(DTime0))]
ETime0 = ETime0[np.invert(np.isnan(ETime0))]
FTime0 = FTime0[np.invert(np.isnan(FTime0))]
GTime0 = GTime0[np.invert(np.isnan(GTime0))]
HTime0 = HTime0[np.invert(np.isnan(HTime0))]
ITime0 = ITime0[np.invert(np.isnan(ITime0))]
JTime0 = JTime0[np.invert(np.isnan(JTime0))]
KTime0 = KTime0[np.invert(np.isnan(KTime0))]
LTime0 = LTime0[np.invert(np.isnan(LTime0))]
MTime0 = MTime0[np.invert(np.isnan(MTime0))]
NTime0 = NTime0[np.invert(np.isnan(NTime0))]

ATime0 = ufloat(np.mean(ATime0), 0)
BTime0 = ufloat(np.mean(BTime0), 0)
CTime0 = ufloat(np.mean(CTime0), 0)
DTime0 = ufloat(np.mean(DTime0), 0)
ETime0 = ufloat(np.mean(ETime0), 0)
FTime0 = ufloat(np.mean(FTime0), 0)
GTime0 = ufloat(np.mean(GTime0), 0)
HTime0 = ufloat(np.mean(HTime0), 0)
ITime0 = ufloat(np.mean(ITime0), 0)
JTime0 = ufloat(np.mean(JTime0), 0)
KTime0 = ufloat(np.mean(KTime0), 0)
LTime0 = ufloat(np.mean(LTime0), 0)
MTime0 = ufloat(np.mean(MTime0), 0)
NTime0 = ufloat(np.mean(NTime0), 0)

ATimeUp = ufloat(np.mean(ATimeUp), np.std(ATimeUp)/np.sqrt(len(ATimeUp)))
BTimeUp = ufloat(np.mean(BTimeUp), np.std(BTimeUp)/np.sqrt(len(BTimeUp)))
CTimeUp = ufloat(np.mean(CTimeUp), np.std(CTimeUp)/np.sqrt(len(CTimeUp)))
DTimeUp = ufloat(np.mean(DTimeUp), np.std(DTimeUp)/np.sqrt(len(DTimeUp)))
ETimeUp = ufloat(np.mean(ETimeUp), np.std(ETimeUp)/np.sqrt(len(ETimeUp)))
FTimeUp = ufloat(np.mean(FTimeUp), np.std(FTimeUp)/np.sqrt(len(FTimeUp)))
GTimeUp = ufloat(np.mean(GTimeUp), np.std(GTimeUp)/np.sqrt(len(GTimeUp)))
HTimeUp = ufloat(np.mean(HTimeUp), np.std(HTimeUp)/np.sqrt(len(HTimeUp)))
ITimeUp = ufloat(np.mean(ITimeUp), np.std(ITimeUp)/np.sqrt(len(ITimeUp)))
JTimeUp = ufloat(np.mean(JTimeUp), np.std(JTimeUp)/np.sqrt(len(JTimeUp)))
KTimeUp = ufloat(np.mean(KTimeUp), np.std(KTimeUp)/np.sqrt(len(KTimeUp)))
LTimeUp = ufloat(np.mean(LTimeUp), np.std(LTimeUp)/np.sqrt(len(LTimeUp)))
MTimeUp = ufloat(np.mean(MTimeUp), np.std(MTimeUp)/np.sqrt(len(MTimeUp)))
NTimeUp = ufloat(np.mean(NTimeUp), np.std(NTimeUp)/np.sqrt(len(NTimeUp)))



ATimeDown = ufloat(np.mean(ATimeDown), np.std(ATimeDown)/np.sqrt(len(ATimeDown)))
BTimeDown = ufloat(np.mean(BTimeDown), np.std(BTimeDown)/np.sqrt(len(BTimeDown)))
CTimeDown = ufloat(np.mean(CTimeDown), np.std(CTimeDown)/np.sqrt(len(CTimeDown)))
DTimeDown = ufloat(np.mean(DTimeDown), np.std(DTimeDown)/np.sqrt(len(DTimeDown)))
ETimeDown = ufloat(np.mean(ETimeDown), np.std(ETimeDown)/np.sqrt(len(ETimeDown)))
FTimeDown = ufloat(np.mean(FTimeDown), np.std(FTimeDown)/np.sqrt(len(FTimeDown)))
GTimeDown = ufloat(np.mean(GTimeDown), np.std(GTimeDown)/np.sqrt(len(GTimeDown)))
HTimeDown = ufloat(np.mean(HTimeDown), np.std(HTimeDown)/np.sqrt(len(HTimeDown)))
ITimeDown = ufloat(np.mean(ITimeDown), np.std(ITimeDown)/np.sqrt(len(ITimeDown)))
JTimeDown = ufloat(np.mean(JTimeDown), np.std(JTimeDown)/np.sqrt(len(JTimeDown)))
KTimeDown = ufloat(np.mean(KTimeDown), np.std(KTimeDown)/np.sqrt(len(KTimeDown)))
LTimeDown = ufloat(np.mean(LTimeDown), np.std(LTimeDown)/np.sqrt(len(LTimeDown)))
MTimeDown = ufloat(np.mean(MTimeDown), np.std(MTimeDown)/np.sqrt(len(MTimeDown)))
NTimeDown = ufloat(np.mean(NTimeDown), np.std(NTimeDown)/np.sqrt(len(NTimeDown)))



table = [ [ATime0.nominal_value, ATimeUp.nominal_value, ATimeUp.std_dev, ATimeDown.nominal_value, ATimeDown.std_dev],
          [BTime0.nominal_value, BTimeUp.nominal_value, BTimeUp.std_dev, BTimeDown.nominal_value, BTimeDown.std_dev],
          [CTime0.nominal_value, CTimeUp.nominal_value, CTimeUp.std_dev, CTimeDown.nominal_value, CTimeDown.std_dev],
          [DTime0.nominal_value, DTimeUp.nominal_value, DTimeUp.std_dev, DTimeDown.nominal_value, DTimeDown.std_dev],
          [ETime0.nominal_value, ETimeUp.nominal_value, ETimeUp.std_dev, ETimeDown.nominal_value, ETimeDown.std_dev],
          [FTime0.nominal_value, FTimeUp.nominal_value, FTimeUp.std_dev, FTimeDown.nominal_value, FTimeDown.std_dev],
          [GTime0.nominal_value, GTimeUp.nominal_value, GTimeUp.std_dev, GTimeDown.nominal_value, GTimeDown.std_dev],
          [HTime0.nominal_value, HTimeUp.nominal_value, HTimeUp.std_dev, HTimeDown.nominal_value, HTimeDown.std_dev],
          [ITime0.nominal_value, ITimeUp.nominal_value, ITimeUp.std_dev, ITimeDown.nominal_value, ITimeDown.std_dev],
          [JTime0.nominal_value, JTimeUp.nominal_value, JTimeUp.std_dev, JTimeDown.nominal_value, JTimeDown.std_dev],
          [KTime0.nominal_value, KTimeUp.nominal_value, KTimeUp.std_dev, KTimeDown.nominal_value, KTimeDown.std_dev],
          [LTime0.nominal_value, LTimeUp.nominal_value, LTimeUp.std_dev, LTimeDown.nominal_value, LTimeDown.std_dev],
          [MTime0.nominal_value, MTimeUp.nominal_value, MTimeUp.std_dev, MTimeDown.nominal_value, MTimeDown.std_dev],
          [NTime0.nominal_value, NTimeUp.nominal_value, NTimeUp.std_dev, NTimeDown.nominal_value, NTimeDown.std_dev],
]



# write it
with open('Mittelwerte.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter = ' ')
    [writer.writerow(r) for r in table]



Time0, TimeUpN, TimeUpS, TimeDownN, TimeDownS = np.loadtxt('Mittelwerte.csv', unpack = True)

TimeUp = un.uarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,])
for i in range(0,len(TimeUpN)):
   TimeUp[i] = ufloat(TimeUpN[i], TimeUpS[i])

TimeDown = un.uarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,])
for i in range(0,len(TimeDownN)):
   TimeDown[i] = ufloat(TimeDownN[i], TimeDownS[i])




##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
### Geschwindigkeiten ausrechnen
s = 5*0.5*0.001


Vel0 = s / Time0       # v_0
VelUpAll = s / TimeUp     # v_auf
VelDownAll = s / TimeDown # v_ab

Vel0Exp = 0.5 * (VelDownAll - VelUpAll)


VelAbw = (Vel0 - Vel0Exp) / Vel0


print('Abweichung des gewollten v_0 vom gemessenen v_0:')
print(VelAbw)


# Mehr als 50% Abweichung haben Anton, Emil, Julius, Kaufmann und Nordpol



VelUp = np.array([VelUpAll[1], VelUpAll[2], VelUpAll[3], VelUpAll[5], VelUpAll[6], VelUpAll[7], VelUpAll[8], VelUpAll[11], VelUpAll[12]])
VelDown = np.array([VelDownAll[1], VelDownAll[2], VelDownAll[3], VelDownAll[5], VelDownAll[6], VelDownAll[7], VelDownAll[8], VelDownAll[11], VelDownAll[12]])



##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
### Konstanten definieren


# Widerstände
R_ABC = 1.69 * 10**6
R_D = 1.66 * 10**6
R_EFGH = 1.65 * 10**6
R_IJK = 1.64 * 10**6
R_LMN = 1.64 * 10**6
R = np.array([R_ABC, R_ABC, R_ABC, R_D, R_EFGH, R_EFGH, R_EFGH, R_EFGH, R_IJK, R_IJK, R_IJK, R_LMN, R_LMN, R_LMN])


# Mit der Tabelle folgen daraus die Temperaturen
T_ABC = 32
T_D = 33
T_EFGH = 34.5
T_IJK = 34
T_LMN = 34
# T = np.array([T_ABC, T_ABC, T_ABC, T_D, T_EFGH, T_EFGH, T_EFGH, T_EFGH, T_IJK, T_IJK, T_IJK, T_LMN, T_LMN, T_LMN])
T = np.array([T_ABC, T_ABC, T_D, T_EFGH, T_EFGH, T_EFGH, T_IJK, T_LMN, T_LMN])
T = T + 273.15

# Aus Abbildung 3 folgt dann die Viskosität der Luft
m = ( 1.88 - 1.81 )*10**(-5) / ( 32 - 17 )
t = 1.81*10**(-5) - m * 17

etaL = m * (T-273.15) + t

# Spannungen
U_ABCD = 250
U_EFGH = 280
U_IJK = 290
U_LMN = 300

# U = np.array([U_ABCD, U_ABCD, U_ABCD, U_ABCD, U_EFGH, U_EFGH, U_EFGH, U_EFGH, U_IJK, U_IJK, U_IJK, U_LMN, U_LMN, U_LMN])
U = np.array([U_ABCD, U_ABCD, U_ABCD, U_EFGH, U_EFGH, U_EFGH, U_IJK, U_LMN, U_LMN])


# Weitere Konstanten
d = ufloat(7.6250*0.001, 0.0051*0.001)       # Durchmesser des Kondensators
B = 6.17 * 10**(-3) * 10**(-2) * 101325/760  # Meter * Newton/Quadratmeter # Konstante für Cunningham-Korrektur
p = 101325                                   # Normaldruck
rhoO = 886                                   # Dichte Öl
R_S = 287.058                                # spezifische Gaskonstante für trockene Luft
g = 9.807                                    # Erdbeschleunigung

rhoL = p / R_S / T                           # Dichte von Luft




##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
### Radius, Ladung, Cunningham-Korrektur

R = un.sqrt(9/4 * etaL/g * (VelDown-VelUp)/(rhoO-rhoL))     # Radius

q = 3*np.pi*etaL * un.sqrt( 9/4 * etaL/g * (VelDown-VelUp)/(rhoO-rhoL) ) * (VelDown-VelUp) * d / U   # Ladung

# qEff = q * (un.sqrt( 1 / (1+B/p/R) ) )**3                 # Korrigierte Ladung mit Cunningham-Eta
qEff = q * (un.sqrt( 1+B/p/R ) )**3

X = un.nominal_values(R)
Yerr = un.std_devs(q)
Y = un.nominal_values(q)
YEff = un.nominal_values(qEff)
YEfferr = un.std_devs(qEff)
#sx = plt.subplots(1, 1)




fig, ax = plt.subplots(1, 1)
minor_locator = AutoMinorLocator(7)
ax.yaxis.set_minor_locator(minor_locator)
ax.grid(b=True, which='major', color='k', linewidth=1.0)
ax.grid(b=True, which='minor', color='b', linewidth=0.8)
plt.errorbar(X, Y, yerr=Yerr, fmt='ko', ecolor='r', capthick=1.0, label = 'Ladungen')
plt.legend(loc='best')

plt.show()




