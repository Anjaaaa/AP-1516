import numpy as np

x = 0.01769#wellenlänge über direkte messung
s_x = 0.0004 #standartabweichung
n = 4 #anzahl messungen

y = 0.0152#wellenlänge über regression herausbekommen
s_y = 0.0005#standartabweichung
m = 10 #anzahl stützpunkte

s_quadrat = ((n-1) * s_x**2 + (m - 1) *s_y**2) / (n+m-2)

t = np.sqrt( n*m / (n+m)) * (x - y)/np.sqrt(s_quadrat)

print (s_quadrat)
print(t)
