import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp
from table import (
        make_table,
        make_SI,
        write)

### Reihenfolge der Farben: violett1, violett2, blau, türkis, grün, gelb, orange, rot

Winkel_Prisma_1, Winkel_Prisma_2, Winkel_Farben_1, Winkel_Farben_2 = np.genfromtxt('Werte.txt', unpack = True)
Winkel_Prisma = Winkel_Prisma_2 - Winkel_Prisma_1
Winkel_Prisma = Winkel_Prisma[np.invert(np.isnan(Winkel_Prisma))]

Winkel_Farben = 360 + Winkel_Farben_1 - Winkel_Farben_2

write('build/Messwerte.tex', make_table([Winkel_Farben_1, Winkel_Farben_2, Winkel_Farben, Winkel_Prisma_1, Winkel_Prisma_2],[2,2,2,2,0]))


#write('build/Tabelle3.tex', make_table([Anfang, Ende, Anfang_2, Ende_2, Impuls, Wellenlange],[2,2,2,2,0,1]))
#write('build/Wellenlange_gesamt.tex', make_SI(Wellenlange_gesamt,r'\meter','e-9', figures=2))
#write('build/Brechungsindex_CO2.tex', make_SI(ufloat(Mittel_n_CO2, Fehler_n_CO2), r'' ,figures=1))
