import numpy as np

x = 0.01769#wellenlänge über direkte messung
s_x = 0.0004 #standartabweichung
n = 4 #anzahl messungen

y = 0.01737#wellenlänge über regression herausbekommen
s_y = 0.00007#standartabweichung
m = 10 #anzahl stützpunkte

s_quadrat = ((n-1) * s_x**2 + (m - 1) *s_y**2) / (n+m-2)

t = np.sqrt( n*m / (n+m)) * (x - y)/np.sqrt(s_quadrat)

print('Wellenlänge x', x, s_x)
print('Wellenlänge y', y, s_y)
print('s:', s_quadrat)
print('t:', t)
