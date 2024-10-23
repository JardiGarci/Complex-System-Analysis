import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.linear_model import LinearRegression

class boxcounting_2D():    

    def __init__(self, img, show = False) -> None:
        self.img = np.array(img)
        self.x, self.y = img.shape
        if show == True:
            plt.imshow(self.img,"gray")
            
    def img_to_boxes(self, s, show = True):
        """
        Counts the boxes of a certain size (s) that have pixels with value 0
        The option show = true shows the image segmented into boxes and the value of the number of boxes, 
        otherwise the function will return the number of boxes N
        """
        new_img = np.array(self.img)
        mesh = np.meshgrid(np.arange(0,self.x,s),np.arange(0,self.y,s))
        N = 0
        for x,y in zip(mesh[0].flatten(), mesh[1].flatten()):  # Loop through each box of size s within the image
            box = self.img[x:x+s,y:y+s]
            if 0 in box: # Counts and marks the boxes with pixel presence = 0
                N += 1
                new_img = cv2.rectangle(new_img, (y, x), (y+s, x+s), (0, 255, 0), 0)   
        if show ==True: # Display the image divided into boxes of a size s
            plt.figure()
            plt.imshow(new_img, "gray")
            print(N)
        else:
            return N 
    def dimension(self, show = True):
        """
        Calculates the fractal dimension alpha, 
        if show = True a graph of the points generated by the number of boxes according to their size will be displayed, together with the approximation, 
        otherwise only the calculated alpha will be returned
        """
        s_boxes = np.arange(6,np.min([self.x,self.y])/4,2) # Generate box sizes in increments of 2 from 0 to 1/4 width or height (whichever is smaller)
        N_boxes = []
        for i,s in enumerate(s_boxes):
            N_boxes.append(self.img_to_boxes(s=int(s), show=False))

        s_log = np.array(np.log(1/s_boxes)).reshape(-1,1)
        N_log = np.array(np.log(N_boxes)).reshape(-1,1)

        # Training the Linear Regression model
        model = LinearRegression()
        model.fit(s_log,N_log)
        alpha = model.coef_
        if show == True:
            plt.figure()
            plt.plot(s_boxes, N_boxes," .")
            
            x = np.linspace(np.min(s_log),np.max(s_log),100).reshape(-1,1) # Generating instances for plotting the model
            plt.figure()
            plt.plot(s_log, N_log, ".")
            plt.plot(x,model.predict(x), label = "LinearRegression")
            plt.xlabel("s")
            plt.ylabel("N(s)")
            plt.legend()
            print(f" Dimensión : {alpha[0][0]:2.3f}")
        else:
            return alpha[0][0]
