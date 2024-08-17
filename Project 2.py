import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time 

def func(x,m,c):
    return m*x + c
    
N = 10000
print(N)
 #Number of particles
J = 10 #Boundary of infinite square well

############################1D Wells######################################

def inf_one_dwell(N,J,max_t):
    

    
    def a(j):
        if j==-J or j==J:
            arrest_probability = 1
        else:
            arrest_probability = 0
        
        return arrest_probability


    #Loop over N particles 

    k_list = np.zeros(max_t+1) #Empty bins for number of particles that survive up to 99 steps



    for i in range(N):

        Survive = True #Particle state

        n=0 #initial number of steps

        spot = 0 #Initial position of particle

        while Survive==True:

            if n == max_t: #Check if particle has survived 100 steps
                k_list[n] += 1
                Survive = False
                

            

            #Current position of particle

            movement_decider = np.random.randint(0,2) #Randomly decide to move left or right

            n+=1 # increase step count

            if movement_decider == 0: #Move left
                spot -= 1
            elif movement_decider == 1:
                spot += 1
            
            death_decider = np.random.uniform(0,1)
            if a(spot)>death_decider: #Check if particle has hit the wall
                Survive = False
                k_list[n] += 1
    
    return k_list
        
def inf_2d_dwell(N,J,max_t,energy_level):
    

    k_list = np.zeros(max_t+2) #Empty bins for number of particles that survive
    
    if energy_level == 0:
        
        def ground_state_potential(i,j):
            r = np.sqrt(i**2 + j**2)
            if r>= J:
                arrest_probability = 1
            else:
                arrest_probability = 0
            
            return arrest_probability
    
        for i in range(N):
    
            Survive = True
    
            n=0
    
            spot_i = 0
            spot_j = 0
    
            while Survive==True:
    
                if n == max_t:
                    k_list[n] += 1
                    Survive = False
                
                n+=1
    
                decider = np.random.randint(0,4)
    
                if decider == 0:
                    spot_i -= 1
                
                elif decider == 1:
                    spot_i += 1
                
                elif decider == 2:
                    spot_j -= 1
                
                elif decider == 3:
                    spot_j += 1
                
                death_probability = np.random.uniform(0,1)
    
                if ground_state_potential(spot_i,spot_j) > death_probability:
                    Survive = False
                    k_list[n] += 1
    
    elif energy_level == 1:
        
        def two_d_energy_level_one(i,j):
            r = np.sqrt(i**2 + j**2)
            
            if r>= J or i == 0:
                arrest_probability = 1 
            else:
                arrest_probability = 0
            
            return arrest_probability
        
        for i in range(N):
      
        
            spot_i = np.random.randint(-J,J+1)
            spot_j = np.random.randint(-J,J+1)
            
            
            
            n = 0
            Survive = True
            
            while Survive == True:
                
                if n == max_t:
                    k_list[n]+=1 
                    Survive = False
                
                death_decider = np.random.uniform(0,1)
                
                if two_d_energy_level_one(spot_i,spot_j) > death_decider:
                    Survive = False
                    k_list[n]+=1 
                
                n+=1
    
                movement_decider = np.random.randint(0,4)
    
                if movement_decider == 0:
                    spot_i -= 1
                
                elif movement_decider == 1:
                    spot_i += 1
                
                elif movement_decider == 2:
                    spot_j -= 1
                
                elif movement_decider == 3:
                    spot_j += 1
            
    elif energy_level == 2:
        
        def two_d_energy_level_two(i,j):
            r = np.sqrt(i**2 + j**2)
            
            if r>= J or i == 0 or j ==0:
                arrest_probability = 1 
            else:
                arrest_probability = 0
            
            return arrest_probability
        
        for i in range(N):
      
        
            spot_i = np.random.randint(-J,J+1)
            spot_j = np.random.randint(-J,J+1)
            
            
            
            n = 0
            Survive = True
            
            while Survive == True:
                
                if n == max_t:
                    k_list[n]+=1 
                    Survive = False
                
                death_decider = np.random.uniform(0,1)
                
                if two_d_energy_level_one(spot_i,spot_j) > death_decider:
                    Survive = False
                    k_list[n]+=1 
                
                n+=1
    
                movement_decider = np.random.randint(0,4)
    
                if movement_decider == 0:
                    spot_i -= 1
                
                elif movement_decider == 1:
                    spot_i += 1
                
                elif movement_decider == 2:
                    spot_j -= 1
                
                elif movement_decider == 3:
                    spot_j += 1

        
    return k_list



#Varying J 




#1D plot of infinite square well with varying J and N
"""
new_list = []
old_list = np.log(inf_one_dwell(N,J,350))
for i in range(len(old_list)):
    if i%2==0:
        new_list.append(old_list[i])

n_range = np.arange(0,351,2)

popt, pcov = curve_fit(func,n_range[8:90],new_list[8:90])

#lambda_list.append(-popt[0])
#error_list.append(np.sqrt(pcov[0,0]))

plt.plot(n_range[8:90],func(n_range[8:90],popt[0],popt[1]),color='red',label='Linear Regression')
plt.scatter(n_range,new_list,label='Generated Values')
plt.xlabel('n')
plt.ylabel('log(k)')
plt.title(f'Estimating $\lambda$ for 1D Well with N={N} and J={J}')
plt.grid()
plt.legend()



print(-popt[0]*J**2)
print(np.sqrt(pcov[0,0]))

plt.show()
"""

"""

#2D plot
value_list = []
error_list = []


start = time.time()
k_values = inf_2d_dwell(N, J, 350,1)

n_range = np.arange(0,len(k_values),1)

plt.plot(n_range[20:90],np.log(k_values)[20:90])
plt.pause(0.01)
popt, pcov = curve_fit(func,n_range[20:90],np.log(k_values)[20:90])

value_list.append(-popt[0]*J**2)
error_list.append(-pcov[0,0]*J**2)


    
print(value_list)
print(error_list)
print(np.mean(value_list))
end = time.time()
print(f' Time = {end - start}')
"""


#Optimizing values

time_taken = []

true_value = (np.pi)**2 / 8

error_list = []

std_list = []


fig, axs = plt.subplots()
axs2 = axs.twinx()
avg_list = []



#Varying N
I_range = np.arange(4,16 ,2)
N_range = [10000,20000,30000,40000,50000]

#fixed N
N=30000
def I_bound(I):
    
    lin_b = [16,30,50,70,100,130]
    err_b = [6,20,20,40,50,50]
    lin_e = [70,130,260,380,550,650]
    err_e = [15,20,40,40,50,50]
    
    return lin_b[int((I/2))-2], lin_e[int((I/2))-2]


    

# Loop over each N value
for I in I_range:
    print(I)
    #begin measurement of computational process
    start = time.time()
    #approximate lambda J**2 plural
    value_list = []
    
    #run multiple iterations
    
    for run in range(5):
        
        new_list = []
        old_list = np.log(inf_one_dwell(N,I,750))
        
        
        
        
        for i in range(len(old_list)):
            if i%2==0:
                new_list.append(old_list[i])
        
        n_range = np.arange(0,751,2)
        
        
        I_bound_value = I_bound(I)
        
        n = n_range[int(I_bound_value[0]/2) : int(I_bound_value[1]/2)]
        new = new_list[int(I_bound_value[0]/2) : int(I_bound_value[1]/2)]
        
        print(n)
        print(new)
        
        
        popt, pcov = curve_fit(func,n,new,sigma=None,absolute_sigma=False)
        
        #plt.plot(n_range,new_list)
       # plt.plot(n,func(n,popt[0],popt[1]),color='red')
        #plt.show()
       
        value_list.append(-popt[0]*I**2)
        

    end = time.time()
    time_taken.append(end - start)
    
    print(time_taken)
    
    error = np.abs((true_value - np.mean(value_list)))/true_value
    error_list.append(error)
    std = np.std(value_list)/true_value
    std_list.append(std)
    
    
print(error_list)
print(std_list)

l1 = axs2.plot(I_range,time_taken,marker='x',color='C2',label='Time taken',ls='none')
l2 = axs.plot(I_range,error_list,label='Relative Error',ls='none',marker='o')
lz = l1 + l2
labs = [l.get_label() for l in lz]
axs.legend(lz, labs, loc=9)
axs.errorbar(I_range,error_list,yerr=std_list,capsize =10,fmt='none')
axs.set_xlabel('Boundary Value I')
axs2.set_ylabel('Time Taken / s')
axs.set_ylabel(f'Relative Error on $\lambda$')
axs.set_title(f'$\lambda$ Error and time taken for N=30000 over various I')
axs2.set_ylim(None,150)

plt.show()

    
    
    
    
    
"""
I= [4,6,8,10,12,14]
lin_b = [16,30,50,70,100,130]
err_b = [6,20,20,40,50,50]
lin_e = [70,130,260,380,550,650]
err_e = [15,20,40,40,50,50]

plt.errorbar(I,lin_b,yerr = err_b,ls='None',marker='x',capsize=5,color='C0',label='Linear Begin')
plt.errorbar(I,lin_e,yerr=err_e,ls='None',marker='x',capsize=5,color='C2',label='Linear End')
plt.legend()
plt.xlabel('Boundary Value I')
plt.ylabel('Number of Steps n')
plt.title('Linearity Range Variation With Boundary Value I')
plt.grid()
plt.show()


#1D plot with arrows
I = 14
"""
"""
new_list = []
old_list = np.log(inf_one_dwell(30000,I,900))
for i in range(len(old_list)):
    if i%2==0:
        new_list.append(old_list[i])

n_range = np.arange(0,901,2)
plt.plot(n_range,new_list)
#plt.annotate('Linear Start', xy=(50.,6.0), xytext=(35., 5.),
             #arrowprops=dict(facecolor='black', shrink=0.05),
             #)
#plt.annotate('Linear End', xy=(350.,2.2), xytext=(280.,1.2 ),
             #arrowprops=dict(facecolor='black', shrink=0.05),
             #)
plt.xlabel('Arrest Step n')
plt.ylabel(f'ln(k)')
plt.grid()
plt.title(f'1D Infinite Square Well Arrest Distribution N=30000 I={I}')
plt.show()


##popt, pcov = curve_fit(func,n_range,np.log(inf_2d_dwell(N, J, 200, 0))[55:125])
#print(-popt[0]*J**2)
#plt.show()

#lissed = inf1dwell(N,J,100)

#new_list = []

#for i in range(len(lissed)):
    ##if i%2==0:
        #new_list.append(lissed[i])
#n_space = np.arange(0,len(lissed),2)

#plt.scatter(n_space[10:45],np.log(new_list[10:45]))
#plt.show()
"""

           







            

        
    

