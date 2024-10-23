import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def Higuchi_1D(signal, ks = False, show = True):
    """
    The Higuchi method is applied to calculate the fractal dimension of a signal (alpha).
    If show = True, the lengths L(k) calculated against the intervals (k) will be displayed.
    Otherwise, only the alpha value will be returned.

    ks = Array with different values ​​of k intervals
    if ks = False it will generated automatically the range of intervals
    """
    X = np.array(signal)
    N = len(X)
    if ks == False:
        ks = np.arange(1, int(3*N/4))
    Ls = []
    for k in ks: # Go through the different intervals
        L_k_m = []
        for m in range(1,k + 1): 
            ranges = np.arange(m, N + 1,k, dtype=int) - 1  # Generates new Lm series with the x values ​​corresponding to the interval m
            X_m = X[ranges]
            L_m = sumatoria(X_m) * norm(N = N,m = m,k = k) / k   # Suma la longitud de cada curva Lm
            L_k_m.append(L_m)
        L_k = np.mean(L_k_m) # You average the Lm curves to get the length of the Lk curve.
        Ls.append(L_k)
    
    ks_log = np.log(ks)
    Ls_log = np.log(1/np.array(Ls))

    # Training the Linear Regression model
    model = LinearRegression()
    model.fit(ks_log.reshape(-1,1),Ls_log.reshape(-1,1))
    alpha = model.coef_  # Alpha fractal dimension calculation
    if show == True:
        x = np.linspace(np.min(ks_log),np.max(ks_log),100).reshape(-1,1) # Geración de instancias para el ploteo del modelo
        plt.figure()
        plt.scatter(ks_log, Ls_log)
        plt.plot(x,model.predict(x),"r", label = "LinearRegression")
        plt.xlabel("k")
        plt.ylabel("L(k)")
        plt.legend()
        print(f" Dimensión : {alpha[0][0]:2.3f}")
    else:
        return alpha[0][0]

    
    
def sumatoria(signal):
    suma = 0
    for x,xpast in zip(signal[1:],signal[:-1]):
        suma += np.abs(x - xpast)
    return suma

def norm(N,m,k):
    
    return ((N - 1) / np.ceil((N - m)) / k) * k        
        