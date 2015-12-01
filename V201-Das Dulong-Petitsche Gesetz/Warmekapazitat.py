import numpy as np
from uncertainties import ufloat


= np.genfromtxt('.txt', unpack = True)

# Einheiten umrechnen
#
#
#

cgmg = ( c_w*m_y*(T_y-T_m)-c_w*m_x(T_y-T_m) ) / (T_m-T_x)


# erstes Objekt
c_1 = (c_w*m_w+cgmg)*(T_m-T_w)/(m_k*(T_k-T_m))
print('Warmekapazitat erster Stoff:', c_1)


# Fur Diskussion
# a = Ausdehungskoeffizient
# k = Kompressionsmodul
# M = Masse
# p = Dichte

V_1 = M_1/p_1
C_V1 = C_P1 - 9*a_1*k_1*V_1*T
