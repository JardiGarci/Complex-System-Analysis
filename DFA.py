import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn import linear_model
import cv2

def trend_surface(surface , grade = 1, show = False):
    x,y = surface.shape
    Sub_IM = np.array(surface)
    TX, TY = np.meshgrid(range(x),range(y))
    x1, y1, z1 = TX.flatten(), TY.flatten(), Sub_IM.flatten()
    if grade == 1:
        ####
        X_data = np.array([x1, y1]).T
        Y_data = z1

        reg = linear_model.LinearRegression().fit(X_data, Y_data)
        a1 = reg.coef_[0]; a2 = reg.coef_[1]; c = reg.intercept_

        # ZZ = FF(TX, TY, a1, a2, c)
        ZZ = a1*TX + a2*TY + c
        ####
        
    elif grade == 2:
        ###
        x1y1, x1x1, y1y1 = x1*y1, x1*x1, y1*y1
        X_data = np.array([x1, y1, x1y1, x1x1, y1y1]).T  
        Y_data = z1

        reg = linear_model.LinearRegression().fit(X_data, Y_data)
        a1 = reg.coef_[0]; a2 = reg.coef_[1]; a3 = reg.coef_[2]; a4 = reg.coef_[3]; a5 = reg.coef_[4]; c = reg.intercept_

        # ZZ = func(TX, TY, a1, a2, a3, a4, a5, c)
        ZZ = a1*TX + a2*TY + a3*TX*TY + a4*TX*TX + a5*TY*TY + c
        ###
    else: print("\n Orden incorrecto \n")

    if show == True:
        fig = plt.figure()
        ax = plt.axes(projection ='3d')
        ax.plot3D(x1, y1, z1, '.r')
        ax.plot_surface(TX, TY, ZZ)

    return ZZ

def dentred_flutation_function_2D(image, size, mask = [], umbral = 0.85, grade = 2, show = False):
    if len(mask) > 0:
        img_mask = mask
    else:
        img_mask = image
        umbral = np.min(image) - 1
    y,x = image.shape

    X = np.arange(0,x - size,size)
    X2 =  np.arange(x , 0 + size, -size) - size

    Y = np.arange(0,y - size,size) 
    Y2 =  np.arange(y , 0 + size, -size) - size

    TX,TY = np.meshgrid(X,Y)
    TX2,TY2 = np.meshgrid(X2,Y2)

    TX = TX.flatten()
    TY = TY.flatten()

    TX2 = TX2.flatten()
    TY2 = TY2.flatten()

    new_img = np.array(image)
    img = np.array(image)
    F = []
    points = []
    for i,j,u,w in zip( TX, TY, TX2, TY2):
        i2 = i + size
        j2 = j + size
        u2 = u + size
        w2 = w + size

        box = np.array(img[j : j2 , i : i2])
        if np.mean(np.array(img_mask[j : j2 , i : i2])) > umbral:
            new_img = cv2.rectangle(new_img, (i, j), (i2, j2), (255, 255, 255), 1)
            surface = trend_surface(box, grade=grade, show=False)    
            residual_matrix = box - surface
            F.append((np.mean(np.square(residual_matrix)))**(1/2))
            points.append([i,j])

        box = np.array(img[w : w2 , u : u2])
        if np.mean(np.array(img_mask[w : w2 , u : u2])) > umbral:
            new_img = cv2.rectangle(new_img, (u, w), (u2, w2), (255, 255, 255), 1)
            surface = trend_surface(box, grade=grade, show=False)    
            residual_matrix = box - surface
            F.append((np.mean(np.square(residual_matrix)))**(1/2))
            points.append([u,w])

        box = np.array(img[w : w2 , i : i2])
        if np.mean(np.array(img_mask[w : w2 , i : i2])) > umbral:
            new_img = cv2.rectangle(new_img, (i, w), (i2, w2), (255, 255, 255), 1)
            surface = trend_surface(box, grade=grade, show=False)    
            residual_matrix = box - surface
            F.append((np.mean(np.square(residual_matrix)))**(1/2))
            points.append([i,w])

        box = np.array(img[j : j2 , u : u2])
        if np.mean(np.array(img_mask[j : j2 , u : u2])) > umbral:
            new_img = cv2.rectangle(new_img, (u, j), (u2, j2), (255, 255, 255), 1)
            surface = trend_surface(box, grade=grade, show=False)    
            residual_matrix = box - surface
            F.append((np.mean(np.square(residual_matrix)))**(1/2))
            points.append([u,j])


    if show == True:
        plt.figure()
        plt.imshow(new_img)
    return F,points

def multidim_cumsum(a):
        out = a[...,:].cumsum(-1)[...,:]
        for i in range(2,a.ndim+1):
            np.cumsum(out, axis=-i, out=out)
        return out

class MF_DFA_2D():

    def __init__(self, img,
                 mask = [],
                 mean = True,
                 cumsum = True,
                 box_sizes = [6, False],    # If False, max size = min(M,N)/4
                 step_size = 'bineo',       # 'Bineo' or a int
                 grade = 2,
                 threshold = 0.85,   
                 ):
        self.img = img
        self.mask = mask
        self.F_q = []

        if mean == True:
            # Substract mean
            if len(mask) >1:
                media = np.mean(self.img[self.mask == 1])
                self.img = np.array(self.img) - media
                self.img[self.mask != 1] = 0
            else:
                media = np.mean(self.img)
                self.img = np.array(self.img) - media

        if cumsum == True:
            # Accumulated sum
            if len(mask) > 1:
                self.img = multidim_cumsum(self.img) 
                self.img[self.mask != 1] = 0
            else:
                self.img = multidim_cumsum(self.img) 


        
        if box_sizes[1] == False:
            max_size = int(np.min(img.shape) / 4)
        elif box_sizes[1] < 0 :  
            max_size = np.min(img.shape) 
        else:
            max_size = box_sizes[1]

            
        if step_size == 'bineo':
            s = box_sizes[0]
            self.box_sizes = [s]
            while True:
                s = int(s * np.sqrt(np.sqrt(2)) ) + 1
                if s > max_size: break
                self.box_sizes.append(s)
        else:
            self.box_sizes = list(range(box_sizes[0],max_size, step_size))

        
        self.F = []
        self.Points = []
        new_box_sizes = []
        for size in self.box_sizes:
            DFF,points = dentred_flutation_function_2D(image = self.img,
                                              mask = self.mask,
                                              size = size,
                                              umbral = threshold,
                                              grade = grade,
                                              show = False)
            if len(DFF) < abs(box_sizes[1]) and box_sizes[1] < 0:
                break
            new_box_sizes.append(size)
            self.F.append(DFF)
            self.Points.append(points)
        self.box_sizes = new_box_sizes
        self.box_sizes_log = np.log10(self.box_sizes)


    def F_to_spectrum(self, lim_q = [-5,5], dq = 0.25):
        self.Q = np.arange( lim_q[0]-dq, lim_q[1] + 2*dq, dq)
        self.F_q_log = [[] for i in self.Q]
        self.F_q = [[] for i in self.Q]
        for F in self.F:
            for j,q in enumerate(self.Q):
                if q == 0:
                    dentred_fluctuation = np.exp(np.mean(np.log(np.array(F))))
                    self.F_q[j].append(dentred_fluctuation) 
                    self.F_q_log[j].append(np.log10(dentred_fluctuation))
                    
                else:
                    dentred_fluctuation = np.mean(np.array(F)**(q))**(1.0/q)
                    self.F_q[j].append(dentred_fluctuation)
                    self.F_q_log[j].append(np.log10(dentred_fluctuation))

        self.holder = []
        self.tau = []
        for q, F_q in zip(self.Q, self.F_q_log):
            h_q = np.polyfit(self.box_sizes_log, F_q, 1)[0]
            self.holder.append(h_q)
            self.tau.append(q * h_q - 2)
        
        self.a = []
        self.f = []
        for i in range(1 , len(self.Q) -1):
            a = (self.tau[i+1] - self.tau[i-1]) / (2 * dq)
            self.a.append(a)
            f = self.Q[i] * a - self.tau[i]
            self.f.append(f)

    def Features(self):
        C2,C1,C0 = np.polyfit(self.Q,self.tau,2)
        self.features_vals = [self.a[self.f.index(max(self.f))],            # a_star
                              min(self.a),                                  # a_min
                              max(self.a),                                  # a_max
                              max(self.a) - min(self.a),                    # width
                              max(self.f) - min(self.f),                    # height
                              sum([np.linalg.norm(np.array((self.a[i-1],self.f[i-1])) - np.array((self.a[1],self.f[1]))) for i in range(1,len(self.a))]),
                              - C0,                     # Lineal function C0 + C1X + C2X^2 of tau
                              C1,
                              - 2 * C2
                              ]
        self.features_names = ['a_star','a_min','a_max','width','height','length', 'C0' ,'C1','C2']
        
        return  self.features_names, self.features_vals
    
    def Show(self):
        if len(self.F_q) == 0:
            self.F_to_spectrum()

        fig = plt.figure(constrained_layout=False, figsize=[9,7])
        fig.suptitle('MF-DFA')

        gs1 = fig.add_gridspec(nrows=1, ncols=1, left=0.05, right=0.6, hspace=0.0, wspace= 0.5)
        f_ax1 = fig.add_subplot(gs1[:, :])
        f_ax1.grid(True)
        f_ax1.set_xlabel('s')
        f_ax1.set_ylabel('F(s)')
        f_ax1.set_xscale('log')
        f_ax1.set_yscale('log')
        for F_q in self.F_q:
            f_ax1.plot(self.box_sizes, F_q)
  
        gs2 = fig.add_gridspec(nrows=4, ncols=1, left=0.7, right=0.98, hspace=0.00)
        f_ax2 = fig.add_subplot(gs2[0, :])
        f_ax2.grid(True)
        f_ax2.set_ylabel('H(q)')
        f_ax2.scatter(self.Q, self.holder, edgecolors='b', c = 'white', s = 15)

        f_ax3 = fig.add_subplot(gs2[1, :])
        f_ax3.grid(True)
        f_ax3.set_ylabel('τ(q)',)
        # f_ax3.set_xlabel('q')
        f_ax3.scatter(self.Q, self.tau, edgecolors='b', c = 'white', s = 15)

        gs3 = fig.add_gridspec(nrows=4, ncols=1, left=0.7, right=0.98, hspace=0.560)
        f_ax4 = fig.add_subplot(gs3[2:, :])
        f_ax4.grid(True)
        f_ax4.set_ylabel('F(α)')
        f_ax4.set_xlabel('α')
        f_ax4.scatter(self.a,self.f, edgecolors='r', c = 'white', s = 35)
        f_ax4.plot(self.a,self.f, 'r')
        plt.show()