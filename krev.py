
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pykrev as pk

A = pk.read_csv('example_A.csv', column_headers = True)

print(type(A))
#A.formula is a list
print(type(A.formula))
#A intensity, and A.mz are numpy.ndarrays
print(type(A.intensity))
print()
#We can summarise the objects in A
A.summary()

A = A.filter_mz(100,1000) #Note, filtering is not in place (an msTuple is immutable)
#A = A.filter_intensity(1e6,1e7)
A = A.filter_spectral_interference(tol=2)

dbe = pk.double_bond_equivalent(A) # the result is a numpy.ndarray of len(A.formula)

#Here we make a van Krevelen style plot where the y axis represents N/C values, and colour code the points by double bond equivalence
plt.figure()
pk.van_krevelen_plot(A, y_ratio = 'HC',c = dbe,s = 7,cmap = 'plasma') #van_krevele_plot takes any keyword arguments that can be passed to pyplot.scatter()
cbar = plt.colorbar() #add a colour bar
cbar.set_label('Double bond equivalence')
plt.grid(False)
plt.show()