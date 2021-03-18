import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import numpy as np
import math
import matplotlib.patches as patches

# Eulers Integration
def simulate(Δt,z,u):
    z = z + Δt*u
    return z


def control(t,z,theta_ref,kp_kd,l,dz):
    error = theta_ref-z
    # desired acceleration
    a = kp_kd[0]*error[0] + kp_kd[1]*error[1]
    #torque
    torque = a
    #forces
    u = [-torque/(2*l),torque/(2*l)]

    #acceleration
    a = np.dot([-l,l],u)
    #print(a.shape)
    dz[0] = z[1]
    dz[1] = a
    return dz

tf=10
Δt = 0.1  #Time Step
time = np.linspace(0.,tf,int(tf/Δt+1))

#Initial Conditions
z = np.array([0.,0.])
z_log = [np.copy(z)]
u_log = [[0,0]]
dz = np.array([0.,0.])
theta_ref = np.array([(3*math.pi)/4,0.])
kp_kd = np.array([10.,3.])
l = 6
g = 9.8    

for t in time[1:]:
    contrl = control(t,z,theta_ref,kp_kd,l,dz)
    #print(contrl)
    z = simulate(Δt,z,contrl)
    #print(sim)
    z_log.append(np.copy(z))


z_log = np.array(z_log)

print(len(z_log))

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(time,z_log[:,0])
plt.grid()
#print(z_log[:,1])
ax1.set_xlabel('Time')
ax1.set_ylabel('Theta')
ax1.set_title('Theta Vs Time (PD_controller)')
#plt.show()

#print(z_log[2,0])
pos_x1 = np.zeros(len(z_log))
pos_y1 = np.zeros(len(z_log))




#Rotation matrix
for a in range(len(z_log)):
    pos_x1[a] = 3*math.cos(z_log[a,0]) - (-0.5*math.sin(z_log[a,0]))
    pos_y1[a] = -0.5*math.cos(z_log[a,0]) + 3*math.sin(z_log[a,0])

theta = z_log[:,0]
print(pos_x1.shape)
print("Next is y")
print(pos_y1)
'''
pos = np.array([pos_x1, pos_y1])
new_pos = np.transpose(pos)
print(new_pos.shape)
'''
#####################################################
################## Animation ########################
#####################################################

fig, ax = plt.subplots()

#center
plt.plot(0,0,'go')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

patch1 = patches.Rectangle((pos_x1[0],pos_y1[0] ),-6,1,ec='b',fc='r')
ax.add_patch(patch1)

q_x = [pos_x1[0],-pos_x1[0]]
q_y = [pos_y1[0],pos_y1[0]]
x_direct = [pos_x1[0]*math.cos(0),pos_x1[0]*math.cos(0)]
y_direct = [pos_y1[0]*math.sin(0),pos_y1[0]*math.sin(0)]

ax.quiver(q_x,q_y,y_direct,x_direct)

patch = patches.Rectangle((0,0),0,0,ec='b',fc='none',ls='--')

def init():
    ax.add_patch(patch)
    return patch,

def animate(i):
    
    patch.set_width(-6)
    patch.set_height(1)
    patch.set_xy([pos_x1[i], pos_y1[i]])
    #patch.set_xy([0,0])
    patch.angle = np.rad2deg(theta[i])
    #ax.clear()
    return patch,

anim = animation.FuncAnimation(fig, animate,init_func=init,frames=len(time),interval=150,blit=True)

plt.show()

































