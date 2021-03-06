import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import numpy as np


def simulate(Δt,z,u):
    z = z + Δt*u
    return z


def control(t, z, pos_ref, kp_kd, mass, g, dz):
    error = pos_ref - z
    u = kp_kd[0]*error[0] + kp_kd[1]*error[1] + mass*g
    dz[0] = z[1]
    dz[1] = u/mass-g
    return dz 

tf=10
Δt = 0.1  #Time Step
time = np.linspace(0.,tf,int(tf/Δt+1))
#print(time)

#Initial Conditions
z = np.array([0.,0.])
z_log = [np.copy(z)]
dz = np.array([0.,0.])
pos_ref = np.array([5.,0.])
kp_kd = np.array([10.,3.])
mass = 1
g = 9.8

for t in time[1:]:
    contrl = control(t,z,pos_ref,kp_kd,mass,g,dz)
    #print(contrl)
    z = simulate(Δt,z,contrl)
    #print(sim)
    z_log.append(np.copy(z))
z_log = np.array(z_log)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
print(z_log.shape)
plt.grid()
ax1.plot(time,z_log[:,0])
ax1.set_xlabel('Time')
ax1.set_ylabel('Position')
ax1.set_title('Position Vs Time (PD_controller)')
z_log[:,1]=time
zero = np.zeros((101,2))
#print(zero)
#plt.show()


#####################################################################
#Animation
#####################################################################
fig, ax = plt.subplots()

def animate(t):
    ax.clear()
    
    #Path
    
    plt.plot(zero[:,1],z_log[:,0], 'r--')
 
    #Initial Conditions
    plt.plot(zero[t,1], z_log[t,0], 'bo')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Position')
    ax1.set_title('Position Vs Time (PD_controller)')
    plt.grid()

anim = FuncAnimation(fig, animate, frames=len(time), interval=60)
#anim.save('free_fall.gif', writer='imagemagick')
plt.show();

