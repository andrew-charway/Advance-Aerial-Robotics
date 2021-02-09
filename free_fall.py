import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import numpy as np

def simulate(Δt,z):
    a=-9.8
    p_dot=z[3:]
    p_ddot=np.array([0.,0.,a])
    u = np.concatenate([p_dot,p_ddot])
    z=z + Δt*u
    return z

'''
calculated the time it takes for an object from a height of +10 to hit 
the ground at height 0. s = ut + (1/2)at^2 where 's' is the total distance,
'u' is initial velocity, 't' is time and 'a' is acceleration due to gravity
'''
tf=1.42784
Δt = 0.1  #Time Step
time = np.linspace(0.,tf,int(tf/Δt)+1)

#Initial Conditions
z = np.array([0.,0.,10.,0.,0.,0.])
z_log = [np.copy(z)]

for t in time:
    z = simulate(Δt,z)
    z_log.append(np.copy(z))
z_log = np.array(z_log)

print(z_log)

plt.grid()
plt.plot(z_log[:,0],z_log[:,2])
#plt.show()

#####################################################################
#Animation
####################################################################
fig, ax = plt.subplots()

def animate(t):
    ax.clear()
    
    #Path
    plt.plot(z_log[:,0],z_log[:,2], 'r--')
 
    #Initial Conditions
    plt.plot(z_log[t,0], z_log[t,2], 'bo')
    
    plt.grid()

anim = FuncAnimation(fig, animate, frames=len(time), interval=60)
anim.save('free_fall.gif', writer='imagemagick')






