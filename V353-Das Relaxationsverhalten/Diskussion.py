import numpy as np

RC1 = 0.775
RC2 = 0.807
RC3 = 0.783

RC = (RC1 + RC2 + RC3) /3

print('Mittelwert der Zeitkonstanten', RC)
print('Abweichung 1', (RC1 / RC -1) *100)
print('Abweichung 2', (RC2 / RC -1) *100)
print('Abweichung 3', (RC3 / RC -1) *100)