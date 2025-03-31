import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


###-------------------------------------------------------------------------------------------
###-------------------------------------------------------------------------------------------
### -----  Operations

def cumsum(a): # Performs a cumulative sum
    out = a[...,:].cumsum(-1)[...,:]
    for i in range(2,a.ndim+1):
        np.cumsum(out, axis=-i, out=out)
    return out

def sub_mean(x):  # Average subtraction of a data set
    return x-np.mean(x)

def sub_mean_mask(img , mask):
    img_mask = np.array(img)*mask
    img_mask[mask != 0] = img_mask[mask != 0] - np.mean(img_mask[mask != 0])
    return(img_mask)

def normalizar(x):
    y = x - np.min(x)
    return (y / np.max(y))

def bineo(s_min, s_max, degree=1):
    """
    Generate intervals for analysis based on a specific scaling factor.
    
    N: Maximum number of intervals
    grade: Determines the scaling factor by progressively taking the square root
    s: Starting interval size
    """
    s = s_min
    val = 2
    N_s = []
    for i in range(1, degree + 1):
        val = np.sqrt(val)  # Progressive square root
    while s < s_max:
        N_s.append(s)
        s = int(s * val) + 1  # Generate the next interval
    return np.array(N_s)



###-------------------------------------------------------------------------------------------
###-------------------------------------------------------------------------------------------
### ------ Shows

def Show(Data):
    """
    data = {'method':'DFA','Hurst','Power Spectrum','Higuchi',
            'm': m,
            'c':c,
            'Ls': Ls,
            's_sizes': s_sizes}
    """

    x = np.log(Data['s_sizes'])

    methods = ['DFA','Hurst','Power Spectrum','Higuchi']
    symbol = ['α','Hu','β','D']

    if Data['method'] in methods:
       case = methods.index(Data['method'])
    else:
        print('Invalid Method')
        return

    x_trend = np.linspace(np.min(x), np.max(x), 100)
    y_trend = Data['m'] * x_trend + Data['c']
    plt.figure(figsize=[15,15])
    plt.scatter(Data['s_sizes'], Data['L_s'])
    plt.loglog()
    if len(Data['s_sizes']) < 20: 
        plt.xticks(ticks=Data['s_sizes'], labels=Data['s_sizes'])
    plt.plot(np.exp(x_trend), np.exp(y_trend), "r", label=f"{symbol[case]} = {Data['m']:2.4}")
    plt.xlabel("Log(s)")
    plt.ylabel("Log(L(s))")
    plt.grid(linewidth = 0.3)
    plt.legend()
    plt.show()



def Show_MF(Data, D = 2):

    fig = plt.figure(figsize=[15,15])
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
    ax2 = plt.subplot2grid((3, 3), (0, 2)) # Holder
    ax3 = plt.subplot2grid((3, 3), (1, 2)) # Tau
    ax4 = plt.subplot2grid((3, 3), (2, 0)) # Alpha
    ax5 = plt.subplot2grid((3, 3), (2, 1)) # F_Alpha
    ax6 = plt.subplot2grid((3, 3), (2, 2)) # Spectrum

    for F,q in zip(Data['FF'],Data['Qs']):
        ax1.plot(Data['s_sizes'],F, label = f'{q}')

    ax1.grid()
    ax1.loglog()
    ax1.set_ylabel('F(s)')
    ax1.set_xlabel('s')
    ax1.set_xticks(Data['s_sizes'])
    ax1.set_xticklabels(Data['s_sizes'])
    ax1.legend(fontsize = 6)

    ax2.plot(Data['Qs'], Data['holder'])
    ax2.grid()
    ax2.set_ylabel('h(q)')
    # ax2.set_ylim([-0.1,D+0.1])
    ax2.set_xlabel('q')

    ax3.plot(Data['Qs'], Data['tau'])
    ax3.grid()
    ax3.set_ylabel('τ(q)')
    ax3.set_xlabel('q')

    ax4.plot(Data['Qs'][1:-1], Data['alpha'])
    ax4.grid()
    ax4.set_ylabel('α')
    ax4.set_xlabel('q')

    ax5.plot(Data['Qs'][1:-1], Data['alpha'])
    ax5.grid()
    ax5.set_ylabel('f(α)')
    ax5.set_xlabel('q')

    ax6.plot(Data['alpha'], Data['f_alpha'])
    ax6.grid()
    ax6.set_ylabel('f(α)')
    ax6.set_xlabel('α')
    ax6.set_ylim([-0.1,D+0.1])
    ax6.set_xlim([-0.1,D+1.1])

    plt.tight_layout(pad=5,h_pad=5)
    plt.show()




###-------------------------------------------------------------------------------------------
###-------------------------------------------------------------------------------------------
### ------ Noises


# Generates Gaussian white noise serie (1D)
def white_noise(mean=0, std=0.1, N=1000):
    """
    Creates a Gaussian white noise signal with specified mean and standard deviation.
    
    mean: Mean of the noise.
    std: Standard deviation of the noise.
    N: Number of samples.
    """
    return np.random.normal(loc=mean, scale=std, size=N)

# Generates Gaussian white noise image (2D)
def white_noise_2D(mean=0, std=0.1, shape= [256,256]):
    """
    Creates a Gaussian white noise image with specified mean and standard deviation.
    
    mean: Mean of the noise.
    std: Standard deviation of the noise.
    N: Number of samples.
    """
    return np.random.normal(loc=mean, scale=std, size=shape)

# Generates Brownian noise serie (1D) (cumulative sum of white noise)
def brownian_noise(mean=0, std=0.1, N=1000):
    """
    Generates Brownian noise by computing the cumulative sum of white noise.
    
    mean: Mean of the white noise.
    std: Standard deviation of the white noise.
    N: Number of samples.
    """
    white_noise = np.random.normal(loc=mean, scale=std, size=N)  # Generate white noise
    return np.cumsum(white_noise)  # Compute the cumulative sum

# Generates Brownian noise image (2D) (cumulative sum of white noise image)
def brownian_noise_2D(mean=0, std=0.1, shape = [256,256]):
    """
    Generates Brownian noise by computing the cumulative sum of white noise.
    
    mean: Mean of the white noise.
    std: Standard deviation of the white noise.
    N: Number of samples.
    """
    wh_noise = np.random.normal(loc=mean, scale=std, size=shape)  # Generate white noise

    return cumsum(wh_noise)  # Compute the cumulative sum

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


###-------------------------------------------------------------------------------------------
###-------------------------------------------------------------------------------------------
### ------------ Analysis



## -------------------- Power Spectrum
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
    x = np.log(frequencies[1:N // 2])
    y = np.log(1/np.array(spectrum[1:N // 2]))
    Beta,c = np.polyfit(x, y, 1)

    data = {'method':'Power Spectrum',
            'm': Beta,
            'c': c,
            'L_s':1/np.array(spectrum[1:N // 2]),
            's_sizes': np.array(frequencies[1:N // 2])}
    
    return data


## -------------------- Higuchi
def sum_differences(signal):
    """
    Calculate the sum of absolute differences between consecutive elements in the signal.
    """
    suma = 0
    for x, xpast in zip(signal[1:], signal[:-1]):
        suma += np.abs(x - xpast)
    return suma

def norm(N, m, k):
    """
    Compute the normalization factor for the Higuchi algorithm.
    
    N: Total length of the signal
    m: Current step in the iteration
    k: Interval size
    """
    return ((N - 1) / ( ( ( (N - m) / k ) // 1 ) * k) )

def Higuchi_1D(signal, ks=False):
    """
    Compute the Higuchi fractal dimension of a 1D signal.
    
    signal: Input signal
    ks: Array of interval sizes (if not provided, it will be generated automatically)
    show: If True, plots the relationship between log(k) and log(L(k))
    """
    X = np.array(signal)

    N = len(X)
    if not ks:
        ks = bineo(s_min=1 , s_max=int(N / 4), degree=1)

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

    x = np.log(ks)
    y = np.log(1/np.array(Ls))
    D,c = np.polyfit(x, y, 1)

    data = {'method':'Higuchi',
            'm': D,
            'c': c,
            'L_s':1/np.array(Ls),
            's_sizes': np.array(ks)}
    
    return data


## -------------------- DFA
# ----- 1D
def DFA_1D(signal, s_box=[]):
    """
    Perform Detrended Fluctuation Analysis (DFA) on a 1D signal to estimate its scaling exponent.
    
    signal: Input signal (1D array)
    s_box: Array of box sizes for the analysis (if not provided, it will be generated automatically)
    show: If True, plots the log-log relationship between box size and fluctuation function
    """
    X = sub_mean(signal) # Center the signal by subtracting the mean
    X = np.cumsum(X)  # Compute the cumulative sum (integrated signal)
    N = len(X)
    
    if len(s_box) == 0:
        s_box = bineo(s_min=6,s_max=int(N / 2), degree=2)  # Generate box sizes automatically
    
    F_s = []  # List to store fluctuation variances for each box size
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
        F_s.append(np.sqrt(np.mean(var)))
    
    # Perform a linear regression on the log-log relationship between box sizes and fluctuation function
    x = np.log(s_box)
    y = np.log(F_s)
    alpha,c = np.polyfit(x, y, 1)

    data = {'method':'DFA',
            'm': alpha,
            'c': c,
            'L_s':F_s,
            's_sizes':s_box}

    return data
        
# ----- 2D
def local_trend_2D(surface , degree = 1):

    y,x = surface.shape
    Sub_IM = np.array(surface)
    TX, TY = np.meshgrid(range(x),range(y))
    x1, y1, z1 = TX.flatten(), TY.flatten(), Sub_IM.flatten()
    
    if degree == 1:
        ####
        X_data = np.array([x1, y1]).T
        Y_data = z1

        reg = LinearRegression().fit(X_data, Y_data)
        a1 = reg.coef_[0]; a2 = reg.coef_[1]; c = reg.intercept_

        # ZZ = FF(TX, TY, a1, a2, c)
        ZZ = a1*TX + a2*TY + c
        
        ####
        
    elif degree == 2:
        ###
        x1y1, x1x1, y1y1 = x1*y1, x1*x1, y1*y1
        X_data = np.array([x1, y1, x1y1, x1x1, y1y1]).T  
        Y_data = z1

        reg = LinearRegression().fit(X_data, Y_data)
        a1 = reg.coef_[0]; a2 = reg.coef_[1]; a3 = reg.coef_[2]; a4 = reg.coef_[3]; a5 = reg.coef_[4]; c = reg.intercept_

        # ZZ = func(TX, TY, a1, a2, a3, a4, a5, c)
        ZZ = a1*TX + a2*TY + a3*TX*TY + a4*TX*TX + a5*TY*TY + c
        ###
    else: print("\n Orden incorrecto \n")
    
    return np.array(ZZ)

def dentred_fluctuation(x, trend_degree = 1):
    '''
    The detrended fluctuation function F(v,w,s) of the segment X(v,w) 
    is defined via the sample variance of the residual matrix
    '''
    surface_trend = local_trend_2D(x, degree=trend_degree)
    residual_matrix = x-surface_trend
    return np.sqrt(np.mean(np.square(residual_matrix)))

def fluctuation_functions(img, segment_sizes, trend_degree = 1, bineo_degree = 1):
    y,x = img.shape
    # Determinación del tamaño de las ventanas mediante Bineo
    if (len(segment_sizes) >= 1)&(len(segment_sizes) < 3):
        s_min = segment_sizes[0]
        if len(segment_sizes) == 2:
            s_max = segment_sizes[1]
        else:
            s_max = np.min([x,y])//4
        segment_sizes = bineo(s_min = s_min, s_max = s_max, degree = bineo_degree)
    
    # Determinación de las funciones de fluctuación de cada tamaño de caja
    FF = []
    for s in segment_sizes:
        tx,ty = np.meshgrid(np.arange(stop=x-s , step=s),np.arange(stop=y-s ,step=s))
        Fs = []
        for i,j in zip(tx.flatten(),ty.flatten()):
            box = np.array(img[ j : j+s , i : i+s ]) 
            F = dentred_fluctuation(x=box, trend_degree = trend_degree)
            Fs.append(F)
        FF.append(Fs)

    return FF, segment_sizes   


def fluctuation_functions_mask(img, mask, segment_sizes, trend_degree = 1, bineo_degree = 1):
    y,x = img.shape
    # Determinación del tamaño de las ventanas mediante Bineo
    if (len(segment_sizes) >= 1)&(len(segment_sizes) < 3):
        s_min = segment_sizes[0]
        if len(segment_sizes) == 2:
            s_max = segment_sizes[1]
        else:
            s_max = np.min([x,y])//4
        segment_sizes = bineo(s_min = s_min, s_max = s_max, degree = bineo_degree)
    
    # Determinación de las funciones de fluctuación de cada tamaño de caja
    FF = []
    for s in segment_sizes:
        tx,ty = np.meshgrid(np.arange(stop=x-s , step=s),np.arange(stop=y-s ,step=s))
        Fs = []
        for i,j in zip(tx.flatten(),ty.flatten()):
            box = np.array(img[ j : j+s , i : i+s ]) 
            box_mask = np.array(mask[ j : j+s , i : i+s ])
            if np.mean(box_mask) == 1: 
                F = dentred_fluctuation(x=box, trend_degree = trend_degree)
                Fs.append(F)
        FF.append(Fs)

    return FF, segment_sizes   


def FF_to_spectrum(FF, segment_sizes = [6], Q_limits = [-5,5], dq = 0.25):

    if len(Q_limits) == 2:
        Qs = np.arange(Q_limits[0],Q_limits[1] + dq, dq)
    elif len(Q_limits) > 2:
        Qs = Q_limits
    else:
        print('Invalid qs limits')
        return

    O_functions = []
    Holder = []
    Tau = []
    for i,q in enumerate(Qs):

        # Calculo de Tau y Holder
        O_function = [] # Overall detrended fluctuation

        if q != 0:
            for Fs in FF:
                if len(Fs) != 0:
                    F_q = (np.mean(np.power(Fs,q)))**(1/q)
                    O_function.append(F_q)
        else:
            for Fs in FF:
                if len(Fs) != 0:
                    F_q = np.exp(np.mean(np.log(Fs)))
                    O_function.append(F_q)

        x = np.log(segment_sizes[:len(O_function)])
        y = np.log(O_function)
        h_q,c = np.polyfit(x, y, 1)


        Holder.append(h_q)
        Tau.append((q * h_q - 2))
        O_functions.append(O_function)

    # Calculo de alpha y f(alpha)
    alpha = np.diff(Tau)/np.diff(Qs)
    f_alpha = Qs[:-1]*alpha - Tau[:-1]

    data = {'alpha':np.array(alpha),
            'f_alpha':np.array(f_alpha),
            'holder': np.array(Holder),
            'tau':np.array(Tau),
            'Qs':np.array(Qs),
            'FF':np.array(O_functions),
            's_sizes':np.array(segment_sizes)}

    return data

def DFA_2D(img, segment_sizes = [6], bineo_degree = 1, trend_degree = 1):

    img = sub_mean(img)
    img = cumsum(img)

    FF,segment_sizes = fluctuation_functions(img, segment_sizes = segment_sizes, trend_degree=trend_degree, bineo_degree = bineo_degree)
    O_function = [] # Overall detrended fluctuation
    for Fs in FF:
        F_s = np.sqrt(np.mean(np.power(Fs,2)))
        O_function.append(F_s)
    x = np.log(segment_sizes)
    y = np.log(O_function)

    alpha,c = np.polyfit(x, y, 1)

    data = {'method':'DFA',
            'm': alpha,
            'c':c,
            'L_s':O_function,
            's_sizes':segment_sizes}
    return data

def MF_DFA_2D(img, segment_sizes = [6], Q_limits = [-5,5], dq = 0.25, bineo_degree = 1, trend_degree = 1, show = False):
    img = sub_mean(img)
    img = cumsum(img)
    FF,segment_sizes = fluctuation_functions(img, segment_sizes = segment_sizes, trend_degree=trend_degree, bineo_degree = bineo_degree)
    return FF_to_spectrum(FF=FF, segment_sizes=segment_sizes, Q_limits=Q_limits, dq=dq)

def MF_DFA_2D_mask(img, mask ,segment_sizes = [6], Q_limits = [-5,5], dq = 0.25, bineo_degree = 1, trend_degree = 1, show = False):
    img = sub_mean_mask(img, mask)
    img = cumsum(img)*mask
    FF,segment_sizes = fluctuation_functions_mask(img,mask, segment_sizes = segment_sizes, trend_degree=trend_degree, bineo_degree = bineo_degree)
    return FF_to_spectrum(FF=FF, segment_sizes=segment_sizes, Q_limits=Q_limits, dq=dq)


def MF_to_features(data):
    """ 
    data = {'alpha':np.array(alpha),
        'f_alpha':np.array(f_alpha),
        'holder': np.array(Holder),
        'tau':np.array(Tau),
        'Qs':np.array(Qs),
        'FF':np.arrays(O_functions),
        's_sizes':np.array(segment_sizes)}
    """
    a_max = data['alpha'][0]
    a_min = data['alpha'][-1]
    a_star = data['alpha'][data['f_alpha'] == np.max(data['f_alpha'])][0]
    dif_a = np.abs(np.max(data['alpha']) - np.min(data['alpha']))
    
    dif_L = np.abs(a_star - a_min)
    dif_R = np.abs(a_max - a_star)
    asy_i = (dif_L - dif_R) / (dif_L + dif_R )# Asymetric index

    f_max = data['f_alpha'][0]
    f_min = data['f_alpha'][-1]
    dif_f = np.abs(np.max(data['f_alpha']) - np.min(data['f_alpha']))

    
    a,b,c = np.polyfit(data['Qs'], data['tau'],2)
    
    data = {'a_max':float(a_max),
            'a_min':float(a_min),
            'dif_a':float(dif_a),
            'a_star':float(a_star),
            'dif_L':float(dif_L),
            'dif_R':float(dif_R),
            'asy_i':float(asy_i),
            'f_max':float(f_max),
            'f_min':float(f_min),
            'dif_f':float(dif_f),
            'a':float(a),
            'b':float(b),
            'c':float(c)
            }
    
    return data

## -------------------- Hurst

def Hurst(signal, min_window=8, max_window=None, num_scales=20, show=False):
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
    scales = bineo(s_min=2 ,s_max=N, degree=1)
    
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
    x = np.log(scales)
    y = np.log(R_S)
    Hu, c = np.polyfit(x, y, 1)

    data = {'method':'Hurst',
            'm': Hu,
            'c': -c,
            'L_s':np.array(scales),
            's_sizes': np.array(R_S)}
    
    return data
 