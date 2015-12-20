import numpy as np
import matplotlib.pyplot as plt


# ohne Rauschen

phi1o, int1o = np.genfromtxt('Phase_ohne.txt', unpack = True)     # erste Messung
phi2o, int2o = np.genfromtxt('Phase_ohne2.txt', unpack = True)    # zweite Messung


phi1o = phi1o + 15                  # Offset
phi2o = phi2o - 165


phi2o[3] = phi2o[3] + 360           # Verschiebung, sodass alle phi im Bereich 0-360 liegen
phi2o[4] = phi2o[4] + 360


print(phi1o, phi2o)
print(int1o, int2o)


f = open("Phase_Int_ohne.txt", "w")
f.write( "# Phase Integral" + "\n" + "# Grad in_Sekunden" + "\n")
for x in range(0,5):
    f.write( str(phi1o[x]) + " " + str(int1o[x]) + "\n" +  str(phi2o[x]) + " " + str(int2o[x]) + "\n"  )
f.close()


#############################################################################################################

# mit Rauschen

phi1m, int1m = np.genfromtxt('Phase_mit.txt', unpack = True)     # erste Messung
phi2m, int2m = np.genfromtxt('Phase_mit2.txt', unpack = True)    # zweite Messung


phi1m = phi1m + 15                  # Offset
phi2m = phi2m 


#phi2m[3] = phi2m[3] + 360           # Verschiebung, sodass alle phi im Bereich 0-360 liegen
#phi2m[4] = phi2m[4] + 360


print(phi1m, phi2m)
print(int1m, int2m)


f = open("Phase_Int_mit.txt", "w")
f.write( "# Phase Integral" + "\n" + "# Grad in_Sekunden" + "\n")
for x in range(0,5):
    f.write( str(phi1m[x]) + " " + str(int1m[x]) + "\n" +  str(phi2m[x]) + " " + str(int2m[x]) + "\n"  )
f.close()







plt.plot(phi1o, int1o, 'r.', label = 'erste Messung ohne')
plt.plot(phi2o, int2o, 'b.', label = 'zweite Messung ohne')
plt.legend(loc = 'best')
plt.show()

plt.plot(phi1m, int1m, 'r.', label = 'erste Messung mit')
plt.plot(phi2m, int2m, 'b.', label = 'zweite Messung mit')
plt.legend(loc = 'best')

plt.show()
