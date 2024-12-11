# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Function to generate an array of intervals for scaling
def bineo(N, grade=1, s=2):
    """
    Generate intervals for analysis based on a specific scaling factor.
    
    N: Maximum number of intervals
    grade: Determines the scaling factor by progressively taking the square root
    s: Starting interval size
    """
    val = 2
    N_s = []
    for i in range(1, grade + 1):
        val = np.sqrt(val)  # Progressive square root
    while s < N:
        N_s.append(s)
        s = int(s * val) + 1  # Generate the next interval
    return np.array(N_s)

# Function to calculate the cumulative sum of absolute differences in a signal
def sum_differences(signal):
    """
    Calculate the sum of absolute differences between consecutive elements in the signal.
    """
    suma = 0
    for x, xpast in zip(signal[1:], signal[:-1]):
        suma += np.abs(x - xpast)
    return suma

# Function to normalize based on the Higuchi algorithm
def norm(N, m, k):
    """
    Compute the normalization factor for the Higuchi algorithm.
    
    N: Total length of the signal
    m: Current step in the iteration
    k: Interval size
    """
    return ((N - 1) / ( ( ( (N - m) / k ) // 1 ) * k) )


# Function to calculate the power spectrum and estimate the spectral exponent β
def power_spectrum(signal, fs=1, show=False):
    """
    Calculate the power spectrum of the input signal and estimate the spectral exponent (β).
    
    signal: 1D array-like input signal
    fs: Sampling frequency
    show: If True, plots the power spectrum and the linear fit
    """
    N = len(signal)  # Number of samples

    # Compute the Fourier Transform
    frequencies = np.fft.fftfreq(N, 1/fs)
    fft_signal = np.fft.fft(signal)
    spectrum = (np.abs(fft_signal) ** 2) / N  # Normalize the power spectrum

    # Perform a linear regression on the log-log plot
    x = np.array(np.log(frequencies[1:N // 2])).reshape(-1, 1)
    y = np.array(np.log(spectrum[1:N // 2])).reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    beta = -model.coef_[0][0]  # The slope of the regression line is -β

    if not show:
        return beta
    else:
        # Plot the power spectrum and linear regression
        y_pred = np.exp(model.predict(x))
        plt.figure(figsize=(10, 5))
        plt.plot(frequencies[:N // 2], spectrum[:N // 2])  # Spectrum for positive frequencies only
        plt.plot(frequencies[1:N // 2], y_pred, 'r', label=f'β = {beta:2.2f}')
        plt.title("Power Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power")
        plt.loglog()  # Logarithmic scale for both axes
        plt.grid(True)
        plt.legend()
        plt.show()

# Function to compute the Higuchi fractal dimension
def Higuchi_1D(signal, ks=False, show=False):
    """
    Compute the Higuchi fractal dimension of a 1D signal.
    
    signal: Input signal
    ks: Array of interval sizes (if not provided, it will be generated automatically)
    show: If True, plots the relationship between log(k) and log(L(k))
    """
    X = np.array(signal)

    N = len(X)
    if not ks:
        ks = bineo(N=int(N / 4), grade=1, s=1)

    Ls = []
    for k in ks:
        L_k_m = []
        for m in np.arange(1, k + 1):
            L_m = []
            ranges = np.arange(start = m, stop = N + 1, step=k, dtype=int)  # Select intervals
            X_m = X[ranges - 1]
            L_m = ( sum_differences(X_m) *  norm(N=N, m=m, k=k) ) / k  # Normalize
            L_k_m.append(L_m)
        L_k = np.mean(L_k_m)
        Ls.append(L_k)


    ks_log = np.log(ks)
    Ls_log = np.log(1/np.array(Ls))

    # Linear regression to estimate the fractal dimension
    model = LinearRegression()
    model.fit( ks_log.reshape(-1, 1), Ls_log.reshape(-1, 1) )
    D = model.coef_[0][0]  # Fractal dimension

    if show:
        # Plot log(k) vs log(L(k))
        x = np.linspace(np.min(ks_log), np.max(ks_log), 100).reshape(-1, 1)
        plt.figure()
        plt.scatter(ks_log, Ls_log)
        plt.plot(x, model.predict(x), "r", label=f"D = {D:2.2}")
        plt.xlabel("Log(k)")
        plt.ylabel("Log(L(k))")
        plt.grid(linewidth = 0.3)
        plt.legend()
    else:
        return D


# Function to calculate the Detrended Fluctuation Analysis (DFA)
def DFA_1D(signal, s_box=False, show=False):
    """
    Perform Detrended Fluctuation Analysis (DFA) on a 1D signal to estimate its scaling exponent.
    
    signal: Input signal (1D array)
    s_box: Array of box sizes for the analysis (if not provided, it will be generated automatically)
    show: If True, plots the log-log relationship between box size and fluctuation function
    """
    X = np.array(signal) - np.mean(signal)  # Center the signal by subtracting the mean
    X = np.cumsum(X)  # Compute the cumulative sum (integrated signal)
    N = len(X)
    
    if not s_box:
        s_box = bineo(N=int(N / 4), grade=2, s=6)  # Generate box sizes automatically
    
    var_s = []  # List to store fluctuation variances for each box size
    for s in s_box:
        var = []
        for i in np.arange(start=0, stop=N - s, step=s):  # Divide signal into non-overlapping boxes
            y = np.array(X[i:i + s]).reshape(-1, 1)  # Extract the segment
            x = np.arange(i, i + s).reshape(-1, 1)  # Generate x-axis values
            
            # Fit a linear model to remove the trend
            model = LinearRegression()
            model.fit(x, y)
            y_trend = model.predict(x)  # Trend line
            
            # Calculate the mean squared difference between the signal and the trend
            var.append(np.mean(np.square(y - y_trend)))
        
        # Compute the square root of the mean variance for the current box size
        var_s.append(np.sqrt(np.mean(var)))
    
    # Perform a linear regression on the log-log relationship between box sizes and fluctuation function
    s_box_log = np.log(s_box).reshape(-1, 1)
    var_s_log = np.log(var_s).reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(s_box_log, var_s_log)
    
    if show:
        # Plot the log-log relationship and the fitted trend line
        var_trend = model.predict(s_box_log)
        plt.plot(s_box_log, var_s_log, '.', label="Data")
        plt.plot(s_box_log, var_trend, label="Trend Line")
        plt.grid()
        plt.ylabel('log(Fd(s))')
        plt.xlabel('log(s)')
        plt.title('DFA')
        plt.legend()
        plt.show()
    
    return model.coef_[0][0]  # Return the scaling exponent (Hurst parameter)

# Function to calculate the Hurst exponent using the Rescaled Range (R/S) method
def hurst_exponent(signal, min_window=8, max_window=None, num_scales=20, show=False):
    """
    Calculate the Hurst exponent of a signal using the R/S analysis method.
    
    signal: Input signal (1D array)
    min_window: Minimum window size for analysis
    max_window: Maximum window size for analysis (default is half the signal length)
    num_scales: Number of scales to analyze
    show: If True, plots the log-log relationship between window size and R/S values
    """
    N = len(signal)
    if max_window is None:
        max_window = N // 2  # Default to half the signal length if not provided
    
    # Generate an array of scales (window sizes) using bineo
    scales = bineo(N=N, grade=1)
    
    R_S = []  # List to store rescaled range values for each scale
    for s in scales:
        rescaled_ranges = []
        for i in range(0, N - s, s):  # Divide the signal into windows of size s
            segment = signal[i:i + s]  # Extract the segment
            deviation = segment - np.mean(segment)  # Remove the mean
            cumulative_deviation = np.cumsum(deviation)  # Compute cumulative deviations
            
            # Calculate the range (R) and standard deviation (S) of the segment
            R = np.max(cumulative_deviation) - np.min(cumulative_deviation)
            S = np.std(segment)
            
            # Add R/S value if S > 0
            if S > 0:
                rescaled_ranges.append(R / S)
        
        R_S.append(np.mean(rescaled_ranges))  # Average R/S values for current scale
    
    # Perform linear regression on the log-log relationship between scales and R/S values
    log_scales = np.log(scales)
    log_R_S = np.log(R_S)
    H, intercept = np.polyfit(log_scales, log_R_S, 1)  # Slope corresponds to Hurst exponent
    
    if show:
        # Plot the log-log relationship and the fitted trend line
        plt.plot(log_scales, log_R_S, 'o', label="Data")
        plt.plot(log_scales, H * log_scales + intercept, label=f"Linear Fit: H = {H:.2f}")
        plt.xlabel("log(scale)")
        plt.ylabel("log(R/S)")
        plt.legend()
        plt.grid()
        plt.show()
    
    return H  # Return the Hurst exponent
