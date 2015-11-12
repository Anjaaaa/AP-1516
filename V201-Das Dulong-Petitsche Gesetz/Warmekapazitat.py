import numpy as np
from uncertainties import ufloat


T_B, T_Bw, T_Bm = np.genfromtxt('Werte.txt', unpack = True)


# Wasser
m_x = 251.80
m_y = 317.80
T_x = 21.6
T_y = 99.7
T_m = 61.0
m_w = m_x + m_y
c_w = 4.18			# Joule/g/K

cgmg = ( c_w*m_y*(T_y-T_m)-c_w*m_x*(T_y-T_m) ) / (T_m-T_x)	 # J/K


# Graphit
T_G = 75.0
T_Gw = 23.0
T_Gm = 26.0
m_G = 247.78 - 139.77

c_G = (c_w*m_w+cgmg)*(T_Gm-T_Gw)/(m_G*(T_G-T_Gm))			# J/g/K

print('spezifische Wärmekapazität Graphit:', c_G)


# Blei
m_B = 685.06 - 140.47
c_B = (c_w*m_w+cgmg)*(T_Bm-T_Bw)/(m_B*(T_B-T_Bm))

print('spezifische Wärmekapazität Blei:', c_B)
print('Mittelwert:', ufloat(np.mean(c_B), np.std(c_B)/np.sqrt(len(c_B))))


# Fur Diskussion
# a = Ausdehungskoeffizient	# 1/K
# k = Kompressionsmodul		# N/m^2
# M = Masse			# g/Mol
# p = Dichte			# g/cm^3

a_G = 8*10**(-6)
a_B = 29.0*10**(-6)
k_G = 33*10**9
k_B = 42*10**9
M_G = 12.0
M_B = 207.2
p_G = 2.24*10**6		# g/m^3
p_B = 11.35*10**6


V_G = M_G/p_G			# m^3/Mol
V_B = M_B/p_B
print('Molvolumen Grahpit:', V_G)
print('Molvolumen Blei:', V_B)


C_PG = c_G*M_G			# J/Mol/K
C_PB = c_B*M_B

C_VG = C_PG - 9 * a_G**2 * k_G * V_G * (T_Gm+273.15)
C_VB = C_PB - 9 * a_B**2 * k_B * V_B * (T_Bm+273.15)

C = 3*8.3144598

print('Mittelwert * Molmasse:', ufloat(np.mean(c_B), np.std(c_B)/np.sqrt(len(c_B))) * M_B)
print('Wärmekapazität Graphit:', C_VG)
print('Wärmekapazität Blei:', C_VB)
print('Mittelwert:', ufloat(np.mean(C_VB), np.std(C_VB)/np.sqrt(len(C_VB))))
print('Dulong-Petit:', C)
