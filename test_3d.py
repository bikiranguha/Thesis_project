import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
#from pylab import meshgrid

# function to be plotted
def z_func(a, b):
    return (a - 3) * (a - 3) + (b - 2) * (b - 2)

x1 = np.arange(15.0, 0, -0.1) # x1 >= 0 according to given conditions
x2 = np.arange(-15.0, 1, 0.1) # x2 <= 1 according to given conditions
X1,X2 = meshgrid(x1, x2)
Z = z_func(X1, X2)
# set all values outside condition to nan
Z[X1**2 - X2 - 3 > 0] = np.nan
"""
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X1, X2, Z, rstride=1, cstride=1,vmin=0, vmax=np.nanmax(Z), 
                       cmap=plt.cm.RdBu,linewidth=0, antialiased=False)

ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
ax.view_init(elev=25, azim=-120)
ax.set_ylim(0,4)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
"""