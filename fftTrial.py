#https://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html
from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt
# Number of sample points
N = 600
# sample spacing
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N)
#y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
y = np.sin(50.0 * 2.0*np.pi*x)
yf = fft(y)
#plt.plot(x,y)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))

#plt.plot( 2.0/N * np.abs(yf))
plt.grid()
plt.show()