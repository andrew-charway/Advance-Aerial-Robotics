import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import numpy as np
import math


#Euler Integration
def simulate(Δt, x, u):
    x += Δt * u
    return x

#Simulation Parameters
freefall = True

#mass and gravity
m = 1
g = 9.81
k = 5
b = 0.02
l_res = 5 #length of spring at rest position

#Gains
kp = 15
kd = 3.9


#Desired height and velocity
z_d = 12
vz_d = 0

#z unit vector
e3 = np.array([0,0,1])
tf = 8.0
Δt = 0.01
time = np.linspace(0.0,tf,int(tf/Δt)+1)

#initial conditions
x = np.array([0.0,0.0,10.0,0.0,0.0,0.0])
dx = np.zeros(6)

x_log = [np.copy(x)]
spring = 5
spring_log = [np.copy(spring)]

for t in time[1:]:
    z = x[2]
    vz = x[5]
    f1 = -m*g
    l = z-l_res
    f_t = (-m*g-b*vz-k*l)
    u = (kp*(z_d-z) + kd*(vz_d-vz) + g)
    if(freefall):
        if(z <= 0):
            dx[3:] = np.zeros(3)
            x[3:] = np.zeros(3)
            freefall = False
        else:
            #Dynamics
            dx[:3] = x[3:]
            dx[3:] = f1/m *e3            
    else:
        #Dynamics
        dx[:3] = x[3:]
        dx[3:] = ((f_t/m) + (u/m-g))*e3
        if (vz < 0): 
            freefall = True                
    if(z<=4 and vz < 0):
        spring = spring-0.08#(0.25+z/2)
    elif(z<=5 and vz > 0):
        spring = (0.25+z/2)
    elif(z >= 5 and vz < 0):
        spring = spring-0.018
    elif (z >= 5 and vz > 0):
        spring = spring+0.05
    x = simulate(Δt, x, dx)
    x_log.append(np.copy(x))
    spring_log.append(np.copy(spring))
x_log = np.array(x_log)
spring_log = np.array(spring_log)

print(spring_log[:110])   
#print(x_log)

plt.plot(time,x_log[:,2])
plt.plot(time[[0,-1]], [z_d, z_d], 'r--', label='$z_d$')
plt.title('Time vs Position Graph of Propeller Assisted Jumping Robot')
plt.xlabel('Time(s)')
plt.ylabel('Position')
plt.legend()
plt.grid()
#plt.show()


RotY = lambda θ: [[math.cos(θ), math.sin(θ)], 
         [math.sin(-θ), math.cos(θ)]]

def spring(x,z, θ, l, ns=10, width = 0.2):
   
    
    px = np.arange(ns+2, dtype=np.float) 
    pz = np.power((-1),px) * width / 2
    px[0] = .5
    pz[0] = 0
    px[-1] = 10.5
    pz[-1] = 0

    px -= 0.5
    px /= ns  # between zero and one
    px *= l  # extend or compress to lenght l
    
    # Rotation
    R = RotY(θ + math.pi/2)
    points = np.array([np.dot(R, [pxi, pzi]) for pxi,pzi in zip(px, pz)])
    px, pz = points[:,0], points[:,1]
    
    # Translation
    px+=x
    pz+=z
    
    return px, pz


def draw_spring(x,z, θ, l, ax=plt):
    px, pz = spring(x,z, θ, l)
    ax.plot(px[0], pz[0], 'o')
    plt.grid()
    ax.plot(px, pz)
    #ax.show()
    
'''
draw_spring(0,1, (math.pi)/-4,1)
draw_spring(1,1, 0,.5)
draw_spring(1,1, 0,1.5)
'''
#####################################################################
#Animation
#####################################################################
fig, ax = plt.subplots()

def animate(t):
    ax.clear()
    ax.set_xlim(-2,2)
    ax.set_ylim(-0.5,13)
    
    #Path
    plt.plot(x_log[:,0],x_log[:,2], 'r--')
 
    #Initial Conditions
    #plt.plot(tm_log[t,0], x_log[t,2], 'bo')
    
    x = x_log[t,0]
    z = x_log[t,2]
    l = spring_log[t]
    θ = 0
    draw_spring(x,z,θ,l)
    ax.grid()

anim = FuncAnimation(fig, animate, frames=len(time), interval=1)
#anim.save('free_fall.gif', writer='imagemagick')
plt.grid()
plt.show();




      
        











