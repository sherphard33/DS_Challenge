import numpy as np
#N = number of samples
#n = current sample
#xn = value of the sinal at time n
#k = current frequency (0 Hz to N-1 Hz)
#Xk = Result of the DFT (amplitude and phase)

def DFT(x):
    """Discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)
"""
def FFT(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 32:  # this cutoff should be optimized
        return DFT(x)
    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N / 2] * X_odd,
                               X_even + factor[N / 2:] * X_odd])
"""
t = np.linspace(0, 0.5, 500)
s = np.sin(40 * 2 * np.pi * t) + 0.5 * np.sin(90 * 2 * np.pi * t)
fft = np.fft.fft(s)
#dft = DFT(s)


for i in range(2):
    print(f"Value at index {i}:\t{fft[i + 1]}", f"\nValue at index {fft.size -1 - i}:\t{fft[-1 - i]}")
