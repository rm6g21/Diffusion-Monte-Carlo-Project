# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:12:20 2024

@author: robbi
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as P

from scipy.integrate import dblquad

def calculate_area(j,n):
    # Define the limits of integration
    x_min = -j
    x_max = j
    y_min = lambda x: -((j**n - np.abs(x)**n)**(1/n))
    y_max = lambda x: ((j**n - np.abs(x)**n)**(1/n))

    # Define the integrand function
    def integrand(x, y):

        return 1

    # Perform the double integration
    area, _ = dblquad(integrand, x_min, x_max, y_min, y_max)

    return area




def circle_error(J):

    
    
    
    fig, axs = plt.subplots()
    
    
    x_dat = []
    y_dat = []
    
    count = 0
    
    run = 10000
    
    for i in range(run):
        #possible values
        x = np.random.randint(-J,J+1)
        y = np.random.randint(-J,J+1)
        
        
        #boundary of well --> can change to circle if needed
        if np.abs(x)**3 + np.abs(y)**3 < J**3 :
            count += 1
            x_dat.append(x)
            y_dat.append(y)
    
    x = np.linspace(-J,J,100)
    y = np.linspace(-J,J,100)
    X, Y = np.meshgrid(x,y)
    
    #for plotting
    Z = np.abs(X)**3 + np.abs(Y)**3 - (J**3)
    
    
    
    
        
    axs.scatter(x_dat,y_dat,color='royalblue')
    axs.contour(X,Y,Z,levels=[0],color='black')
    
    #plot approx area
    for i in range(len(x_dat)):
        
        axs.add_patch(P.Rectangle((x_dat[i]-0.5,y_dat[i]-0.5),1,1,color='royalblue'))
    
    axs.set_xlabel('X')
    axs.set_ylabel('Y')
    axs.set_title(f'Comparison of circle approximation for J={J} against true circle')
    
    plt.show()
    
    
    
    ratio = count / run
    print(count)
    approx_area = ratio * (2*J)**2
    true_area = calculate_area(J,3)
    print(f'True area is {true_area}')
    print(f'approx area is {approx_area}')
    
    
    relative_error = (true_area - approx_area) / (true_area)
    return relative_error
    

#obtain relative error data
error_dat = []
rang = np.arange(4,28,4)
for J in rang:
    error_dat.append(circle_error(J))
  
    
plt.scatter(rang,error_dat,marker = 'x',label='Relative Error',color='black')
plt.grid()
plt.xlabel('Value of J')
plt.ylabel('Relative Error')
plt.legend()
plt.title('Relative Error On Approximation of Circle Using Grid of Length 2Jx2J')

print(rang)
print(error_dat)
    
    



        
    


    