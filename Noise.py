# import numpy as np

# def white_noise(mean = 0, std = 0.1, N = 1000):
#     return np.random.normal(loc=mean, scale=std, size=N)

# def brown_noise(mean = 0, std = 0.1, N = 1000):
#     white_noise = np.random.normal(loc=mean, scale=std, size=N)
#     return  np.cumsum(white_noise)

# def Fractional_Gaussian_Noise(beta=-1, mean=0, std=0.1, N=1000, fs=1):
#     # Generate white noise
#     wn = np.random.normal(mean, std, N)
    
#     # Fourier transform
#     Y = np.fft.fft(wn)
    
#     # Create the associated frequencies
#     freqs = np.fft.fftfreq(N, d=1/fs)
#     freqs[0] = 1e-6  # Evitar división por cero para frecuencia 0
    
#     # Adjust amplitudes according to the power spectrum
#     scaling_factors = np.abs(freqs) ** (-beta / 2.0)
#     Y_new = Y * scaling_factors
    
#     # Reconstruct the signal with the Inverse Fourier Transform
#     reconstructed_signal = np.fft.ifft(Y_new).real
    
#     return reconstructed_signal

# def binomial_cascade_1d(N,p = 0.6):
#     """
#     Generates a 1-dimensional multiplicative binomial cascade
#     N = Iteration
#     """
#     mult = np.array([ p , 1-p ])
#     cascade = np.array([1])
#     for _ in range(N):
#         new_cascade = np.array([])
#         for x in cascade:
#             new_cascade = np.append(new_cascade, x*mult)
#         cascade = new_cascade
#     return cascade

# def binomial_cascade_2d(N,p = 0.1, q = 0.2, r = 0.3, s = 0.4):
#     """
#     Generates a 2-dimensional multiplicative binomial cascade
#     N = Iteration
#     """
#     mult = np.array([[p,q],[r,s]])
#     cascade = np.array([[1]])
#     for _ in range(N):
#         X,Y = cascade.shape
#         new_cascade = np.zeros([X*2,Y*2])
#         for x in range(X):
#             for y in range(Y):
#                 opera = cascade[x,y]*mult
#                 new_cascade[x*2:x*2 + 2, y*2 : y*2 + 2] = opera
#         cascade = new_cascade
#     return cascade


import numpy as np

# Generates Gaussian white noise
def white_noise(mean=0, std=0.1, N=1000):
    """
    Creates a Gaussian white noise signal with specified mean and standard deviation.
    
    mean: Mean of the noise.
    std: Standard deviation of the noise.
    N: Number of samples.
    """
    return np.random.normal(loc=mean, scale=std, size=N)

# Generates Brownian noise (cumulative sum of white noise)
def brown_noise(mean=0, std=0.1, N=1000):
    """
    Generates Brownian noise by computing the cumulative sum of white noise.
    
    mean: Mean of the white noise.
    std: Standard deviation of the white noise.
    N: Number of samples.
    """
    white_noise = np.random.normal(loc=mean, scale=std, size=N)  # Generate white noise
    return np.cumsum(white_noise)  # Compute the cumulative sum

# Generates fractional Gaussian noise using Fourier transform
def Fractional_Gaussian_Noise(beta=-1, mean=0, std=0.1, N=1000, fs=1):
    """
    Generates fractional Gaussian noise by adjusting the power spectrum of white noise.
    
    beta: Exponent defining the frequency distribution (fractal scaling).
    mean: Mean of the initial noise.
    std: Standard deviation of the initial noise.
    N: Number of samples.
    fs: Sampling frequency.
    """
    # Generate white noise
    wn = np.random.normal(mean, std, N)
    
    # Compute the Fourier transform of the white noise
    Y = np.fft.fft(wn)
    
    # Compute the associated frequencies
    freqs = np.fft.fftfreq(N, d=1/fs)
    freqs[0] = 1e-6  # Avoid division by zero for frequency 0
    
    # Adjust amplitudes according to the power spectrum
    scaling_factors = np.abs(freqs) ** (-beta / 2.0)
    Y_new = Y * scaling_factors
    
    # Reconstruct the signal using the Inverse Fourier Transform
    reconstructed_signal = np.fft.ifft(Y_new).real
    
    return reconstructed_signal

# Generates a 1D multiplicative binomial cascade
def binomial_cascade_1d(N, p=0.6):
    """
    Generates a 1-dimensional multiplicative binomial cascade.
    
    N: Number of iterations (cascade depth).
    p: Multiplication probability at each stage.
    """
    mult = np.array([p, 1 - p])  # Multiplicative probabilities at each step
    cascade = np.array([1])  # Initial state of the cascade
    for _ in range(N):
        new_cascade = np.array([])  # Array to store the new values
        for x in cascade:
            new_cascade = np.append(new_cascade, x * mult)  # Multiply each value by the probabilities
        cascade = new_cascade  # Update the cascade for the next iteration
    return cascade

# Generates a 2D multiplicative binomial cascade
def binomial_cascade_2d(N, p=0.1, q=0.2, r=0.3, s=0.4):
    """
    Generates a 2-dimensional multiplicative binomial cascade.
    
    N: Number of iterations (cascade depth).
    p, q, r, s: Multiplication probabilities for 2D subdivisions.
    """
    mult = np.array([[p, q], [r, s]])  # Multiplicative probabilities in 2D
    cascade = np.array([[1]])  # Initial state of the cascade
    for _ in range(N):
        X, Y = cascade.shape  # Get the current dimensions of the cascade
        new_cascade = np.zeros([X * 2, Y * 2])  # Create a new matrix for the next iteration
        for x in range(X):
            for y in range(Y):
                # Multiply each cell by the probabilities and expand into a 2x2 matrix
                opera = cascade[x, y] * mult
                new_cascade[x * 2:x * 2 + 2, y * 2:y * 2 + 2] = opera
        cascade = new_cascade  # Update the cascade for the next iteration
    return cascade
