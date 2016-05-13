import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties.umath import *
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






##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
### Geschwindigkeiten ausrechnen
s = 5*0.5*0.001


### v_0

AVel0 = s / ATime0 
BVel0 = s / BTime0
CVel0 = s / CTime0
DVel0 = s / DTime0
EVel0 = s / ETime0
FVel0 = s / FTime0
GVel0 = s / GTime0
HVel0 = s / HTime0
IVel0 = s / ITime0
JVel0 = s / JTime0
KVel0 = s / KTime0
LVel0 = s / LTime0
MVel0 = s / MTime0
NVel0 = s / NTime0

### v_ab

AVelDown = s / ATimeDown 
BVelDown = s / BTimeDown
CVelDown = s / CTimeDown
DVelDown = s / DTimeDown
EVelDown = s / ETimeDown
FVelDown = s / FTimeDown
GVelDown = s / GTimeDown
HVelDown = s / HTimeDown
IVelDown = s / ITimeDown
JVelDown = s / JTimeDown
KVelDown = s / KTimeDown
LVelDown = s / LTimeDown
MVelDown = s / MTimeDown
NVelDown = s / NTimeDown


### v_auf

AVelUp = s / ATimeUp 
BVelUp = s / BTimeUp
CVelUp = s / CTimeUp
DVelUp = s / DTimeUp
EVelUp = s / ETimeUp
FVelUp = s / FTimeUp
GVelUp = s / GTimeUp
HVelUp = s / HTimeUp
IVelUp = s / ITimeUp
JVelUp = s / JTimeUp
KVelUp = s / KTimeUp
LVelUp = s / LTimeUp
MVelUp = s / MTimeUp
NVelUp = s / NTimeUp


### 0.5 * (v_ab - v_auf)

AVel0Exp = 0.5 * ( AVelDown - AVelUp )
BVel0Exp = 0.5 * ( BVelDown - BVelUp ) 
CVel0Exp = 0.5 * ( CVelDown - CVelUp )
DVel0Exp = 0.5 * ( DVelDown - DVelUp )
EVel0Exp = 0.5 * ( EVelDown - EVelUp )
FVel0Exp = 0.5 * ( FVelDown - FVelUp )
GVel0Exp = 0.5 * ( GVelDown - GVelUp )
HVel0Exp = 0.5 * ( HVelDown - HVelUp )
IVel0Exp = 0.5 * ( IVelDown - IVelUp )
JVel0Exp = 0.5 * ( JVelDown - JVelUp )
KVel0Exp = 0.5 * ( KVelDown - KVelUp )
LVel0Exp = 0.5 * ( LVelDown - LVelUp )
MVel0Exp = 0.5 * ( MVelDown - MVelUp )
NVel0Exp = 0.5 * ( NVelDown - NVelUp )


AVel0Exp = ufloat( np.mean(AVel0Exp) , np.std(AVel0Exp)/np.sqrt(len(AVel0Exp)) )
BVel0Exp = ufloat( np.mean(BVel0Exp) , np.std(BVel0Exp)/np.sqrt(len(BVel0Exp)) )
CVel0Exp = ufloat( np.mean(CVel0Exp) , np.std(CVel0Exp)/np.sqrt(len(CVel0Exp)) )
DVel0Exp = ufloat( np.mean(DVel0Exp) , np.std(DVel0Exp)/np.sqrt(len(DVel0Exp)) )
EVel0Exp = ufloat( np.mean(EVel0Exp) , np.std(EVel0Exp)/np.sqrt(len(EVel0Exp)) )
FVel0Exp = ufloat( np.mean(FVel0Exp) , np.std(FVel0Exp)/np.sqrt(len(FVel0Exp)) )
GVel0Exp = ufloat( np.mean(GVel0Exp) , np.std(GVel0Exp)/np.sqrt(len(GVel0Exp)) )
HVel0Exp = ufloat( np.mean(HVel0Exp) , np.std(HVel0Exp)/np.sqrt(len(HVel0Exp)) )
IVel0Exp = ufloat( np.mean(IVel0Exp) , np.std(IVel0Exp)/np.sqrt(len(IVel0Exp)) )
JVel0Exp = ufloat( np.mean(JVel0Exp) , np.std(JVel0Exp)/np.sqrt(len(JVel0Exp)) )
KVel0Exp = ufloat( np.mean(KVel0Exp) , np.std(KVel0Exp)/np.sqrt(len(KVel0Exp)) )
LVel0Exp = ufloat( np.mean(LVel0Exp) , np.std(LVel0Exp)/np.sqrt(len(LVel0Exp)) )
MVel0Exp = ufloat( np.mean(MVel0Exp) , np.std(MVel0Exp)/np.sqrt(len(MVel0Exp)) )
NVel0Exp = ufloat( np.mean(NVel0Exp) , np.std(NVel0Exp)/np.sqrt(len(NVel0Exp)) )


AVel0Abw = ( AVel0 - AVel0Exp ) / AVel0 
BVel0Abw = ( BVel0 - BVel0Exp ) / BVel0
CVel0Abw = ( CVel0 - CVel0Exp ) / CVel0
DVel0Abw = ( DVel0 - DVel0Exp ) / DVel0
EVel0Abw = ( EVel0 - EVel0Exp ) / EVel0
FVel0Abw = ( FVel0 - FVel0Exp ) / FVel0
GVel0Abw = ( GVel0 - GVel0Exp ) / GVel0
HVel0Abw = ( HVel0 - HVel0Exp ) / HVel0
IVel0Abw = ( IVel0 - IVel0Exp ) / IVel0
JVel0Abw = ( JVel0 - JVel0Exp ) / JVel0
KVel0Abw = ( KVel0 - KVel0Exp ) / KVel0
LVel0Abw = ( LVel0 - LVel0Exp ) / LVel0
MVel0Abw = ( MVel0 - MVel0Exp ) / MVel0
NVel0Abw = ( NVel0 - NVel0Exp ) / NVel0



print('Abweichung aus gemessenem des gewollten v_0 vom gemessenen v_0')
print('A:', AVel0Abw)
print('B:', BVel0Abw)
print('C:', CVel0Abw)
print('D:', DVel0Abw)
print('E:', EVel0Abw)
print('F:', FVel0Abw)
print('G:', GVel0Abw)
print('H:', HVel0Abw)
print('I:', IVel0Abw)
print('J:', JVel0Abw)
print('K:', KVel0Abw)
print('L:', LVel0Abw)
print('M:', MVel0Abw)
print('N:', NVel0Abw)



# Anton, Emil, Julius, Kaufmann und Nordpol werden auskommentiert, da ihre Abweichung von v_0 mehr als 50% ist.

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


# Mit der Tabelle folgen daraus die Temperaturen
T_ABC = 32
T_D = 33
T_EFGH = 34.5
T_IJK = 34
T_LMN = 34


# Aus Abbildung 3 folgt dann die Viskosität der Luft
m = ( 1.88 - 1.81 )*10**(-5) / ( 32 - 17 )
t = 1.81*10**(-5)-m * 17

def etaL(T):
   return m*T+t

etaL_ABC = etaL(T_ABC)
etaL_D = etaL(T_D)
etaL_EFGH = etaL(T_EFGH)
etaL_IJK = etaL(T_IJK)
etaL_LMN = etaL(T_LMN)


# Spannungen
U_ABCD = 250
U_EFGH = 280
U_IJK = 290
U_LMN = 300


# Weitere Konstanten
d = ufloat(7.6250*0.001, 0.0051*0.001)       # Durchmesser des Kondensators
B = 6.17 * 10**(-3) * 10**(-2) * 101325/760  # Meter * Newton/Quadratmeter # Konstante für Cunningham-Korrektur
p = 101325                                  # Normaldruck
rhoO = 886                                   # Dichte Öl
R_S = 287.058                                # spezifische Gaskonstante für trockene Luft
g = 9.807                                    # Erdbeschleunigung

def rhoL(T):                                 # Dichte von Luft
   return p / R_S / T



##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
### Radius, Ladung, Cunningham-Korrektur

### Radius

def Radius(VelDown, VelUp, T):
   return ( 9/4 * etaL(T)/g * (VelDown - VelUp)/(rhoO - rhoL(T)) )


# ARadius = Radius(AVelDown, AVelUp, T_ABC)
BRadius = Radius(BVelDown, BVelUp, T_ABC)
CRadius = Radius(CVelDown, CVelUp, T_ABC)
DRadius = Radius(DVelDown, DVelUp, T_D)
# ERadius = Radius(EVelDown, EVelUp, T_EFGH)
FRadius = Radius(FVelDown, FVelUp, T_EFGH)
GRadius = Radius(GVelDown, GVelUp, T_EFGH)
HRadius = Radius(HVelDown, HVelUp, T_EFGH)
IRadius = Radius(IVelDown, IVelUp, T_IJK)
# JRadius = Radius(JVelDown, JVelUp, T_IJK)
# KRadius = Radius(KVelDown, KVelUp, T_IJK)
LRadius = Radius(LVelDown, LVelUp, T_LMN)
MRadius = Radius(MVelDown, MVelUp, T_LMN)
# NRadius = Radius(NVelDown, NVelUp, T_LMN)

# DEr Radius von N ist bei einem Werte negativ

### Cunningham-Korrektur

def etaEff(T, Radius):
   return ( etaL(T) * ( 1 / ( 1 + B/Radius/p ) ) )



### Ladung

def q(T, VelDown, VelUp, U, Radius):
   a = (VelDown-VelUp)/(rhoO-rhoL(T))
   b = (VelDown-VelUp) * d / U
   return ( 3*np.pi*etaEff(T,Radius) * np.sqrt( 9/4 * etaEff(T,Radius)/g * a ) * b )



# Aq = q(T_ABC, AVelDown, AVelUp, U_ABCD, ARadius)
Bq = q(T_ABC, BVelDown, BVelUp, U_ABCD, BRadius)
Cq = q(T_ABC, CVelDown, CVelUp, U_ABCD, CRadius)
Dq = q(T_D, DVelDown, DVelUp, U_ABCD, DRadius)
# Eq = q(T_EFGH, EVelDown, EVelUp, U_EFGH, ERadius)
Fq = q(T_EFGH, FVelDown, FVelUp, U_EFGH, FRadius)
Gq = q(T_EFGH, GVelDown, GVelUp, U_EFGH, GRadius)
Hq = q(T_EFGH, HVelDown, HVelUp, U_EFGH, HRadius)
Iq = q(T_IJK, IVelDown, IVelUp, U_IJK, IRadius)
# Jq = q(T_IJK, JVelDown, JVelUp, U_IJK, JRadius)
# Kq = q(T_IJK, KVelDown, KVelUp, U_IJK, KRadius)
Lq = q(T_LMN, LVelDown, LVelUp, U_LMN, LRadius)
Mq = q(T_LMN, MVelDown, MVelUp, U_LMN, MRadius)
# Nq = q(T_LMN, NVelDown, NVelUp, U_LMN, NRadius)


def qEff(q, Radius):
   return ( q * (np.sqrt( 1+B/p/Radius ))**3 )



# AqEff = qEff(Aq, ARadius)
BqEff = qEff(Bq, BRadius)
CqEff = qEff(Cq, CRadius)
DqEff = qEff(Dq, DRadius)
# EqEff = qEff(Eq, ERadius)
FqEff = qEff(Fq, FRadius)
GqEff = qEff(Gq, GRadius)
HqEff = qEff(Hq, HRadius)
IqEff = qEff(Iq, IRadius)
# JqEff = qEff(Jq, JRadius)
# KqEff = qEff(Kq, KRadius)
LqEff = qEff(Lq, LRadius)
MqEff = qEff(Mq, MRadius)
# NqEff = qEff(Nq, NRadius)





