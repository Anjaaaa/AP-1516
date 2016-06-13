import numpy as np
from table import(
	make_table,
	make_SI,
	write,
	make_composed_table)



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

write('build/tabelle_WerteA.txt', make_table([e1*100, s1*100, (s1-np.mean(s1-e1))*100, e2*100, s2*100, (s2-np.mean(s2-e2))*100], [3,3,3,3,3,3]))
