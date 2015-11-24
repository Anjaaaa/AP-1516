import numpy as np

# Wärmestrom Temp.2-Temp.1 Messing
x = 0.03
t = np.array([50, 100, 150, 200, 250, 300])
T = np.array([4+2.35/2.15, 4+2.35/1.4, 3+2.35/2.25, 3+2.35/1.15, 3+2.35/0.45, 2+2.35/2.3])
k = 120
A = 0.012*0.004

Stromdichte = -k*A*(T/t)

print('Stromdichte:', Stromdichte)



# Phasendifferenz Messing 1, 2
# 2.4 cm = 100 s  ->  1 cm = 100/2.4
# 1.65 cm = 5°  ->  1 cm = 5/1.65
Delta_t = np.array([0.2, 0.15, 0.15, 0.2, 0.2, 0.2, 0.2, 0.2])	# Abstand der Minima in cm
Delta_t = Delta_t*100 / 2.4
A_klein = np.array([1.7, 1.7, 1.75, 1.95, 1.85, 1.7, 2, 2])
A_klein = A_klein*5 / 1.65
A_groß = np.array([2.6, 2.8, 2.85, 2.75, 2.7, 2.35, 2.7, 2.5])
A_groß = A_groß*5 / 1.65
A = (A_klein+A_groß)/4

print(Delta_t)
print(A)
