import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat

# Spannungen und Ströme einlesen
spannung_orange, orange = np.genfromtxt('Daten1.txt', unpack=True)
spannung, rot, grun, lila, blau, uv = np.genfromtxt('Daten2.txt', unpack=True)
matrix=np.array([rot, grun, lila, blau, uv])

#pikoampere in ampere umrechnen
orange = orange*10**-12
matrix = matrix*10**-12




#Werte mit positivem strom heraussuchen und plotten
matrix_plus=matrix
for j in range(0,5):
	for i in range (0,len(matrix[j,:]-1)):
		if (matrix[j,i] < 0):
			matrix_plus[j,i] = np.nan


y = matrix_plus[0]
mask = np.isfinite(y)


plt.plot(spannung, np.sqrt(matrix_plus[0,:]), 'ro', label='rot')
plt.plot(spannung, np.sqrt(matrix_plus[1,:]), 'go', label='grün')
plt.plot(spannung, np.sqrt(matrix_plus[2,:]), 'mo', label='lila')
plt.plot(spannung, np.sqrt(matrix_plus[3,:]), 'bo', label='blau')
plt.plot(spannung, np.sqrt(matrix_plus[4,:]), 'ko', label='UV')

plt.xlim(-2,2)
plt.ylabel(r'Wurzel des Stromes / $\sqrt{\mathrm{I}}$')
plt.xlabel(r'Spannung / V')
plt.legend(loc='best')
plt.savefig('build/AlleWellenlangen.png')
plt.show()
		
	
orange_plus = orange
for i in range (0,len(orange)-1):
	if (orange[i]< 0):
		orange_plus[i] = np.nan
mask=np.logical_not(np.isnan(orange_plus))
print(orange_plus[mask])
	


plt.plot(spannung_orange, np.sqrt(orange), 'bx')
plt.xlim(-20,20)
plt.ylabel(r'Wurzel des Stromes / $\sqrt{\mathrm{I}}$')
plt.xlabel(r'Spannung / V')
plt.savefig('build/OrangeWellenlange.png')
plt.show()

#lineare regression, um den x-achsenabschnitt zu bestimmen
#orange ist linear im 11 bis 24. element


orange_plus2=np.sqrt(orange_plus[11:24])
spannung_orange2 = spannung_orange[11:24]
def  g(spannung_orange, m, b):
    return m*(spannung_orange) + b

parameters, popt = curve_fit(g, spannung_orange[mask], orange_plus[mask])
m = ufloat(parameters[0], np.sqrt(popt[0,0]))
b = ufloat(parameters[1], np.sqrt(popt[1,1]))

x = np.linspace(-5, 5)
#plt.errorbar(spannung_orange, np.sqrt(orange_plus), xerr=1, yerr=unp.std_devs(phi_gesamt), fmt='r.')
plt.plot(x, g(x, *parameters), 'b-')
plt.plot(spannung_orange, orange_plus[mask], 'b*')
plt.xlim(-5,5)
plt.ylabel(r'Wurzel des Stromes / $\sqrt{\mathrm{I}}$')
plt.xlabel(r'Spannung / V')
plt.savefig('build/regression_orange.png')
plt.show()

#print(popt)
#print(-b/m)
#print(spannung_orange2, orange_plus2)
print(matrix_plus)
print(spannung)

#regession für alle matrixelemente
def  g(spannung, m, b):
    return m*(spannung) + b

parameters, popt = curve_fit(g, spannung[mask], y[mask])

#print(popt)
x = np.linspace(-2, 2)
#plt.errorbar(spannung_orange, np.sqrt(orange_plus), xerr=1, yerr=unp.std_devs(phi_gesamt), fmt='r.')
plt.plot(x, g(x, *parameters), 'b-')
plt.plot(spannung, matrix_plus[0,:], 'b*')
plt.xlim(-5,5)
plt.ylabel(r'Wurzel des Stromes / $\sqrt{\mathrm{I}}$')
plt.xlabel(r'Spannung / V')
plt.savefig('build/regression_rot.png')
plt.show()








 	
	