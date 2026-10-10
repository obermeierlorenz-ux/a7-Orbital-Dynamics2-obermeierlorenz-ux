import numpy as np  
import matplotlib.pyplot as plt
from body import Body
from simulation import (move_to_center_of_mass_frame, vel_verlet,
                        total_momentum, center_of_mass_position)


m1 = 1.0
m2 = 2.0
d = 1.0
G = 1.0

r1_0 = np.array([0.0, 0.0])
r2_0 = np.array([1.0, 0.0])

v1_0 = np.array([0.0,-1.4])
v2_0 = np.array([0.0, 0.0])

body1 = Body(mass=m1, position = r1_0, velocity = v1_0)
body2 = Body(mass=m2, position = r2_0, velocity = v2_0)

bodies = [body1, body2]


v_0 = np.linalg.norm(v2_0 - v1_0)# gives me one number v_0
r_0 = np.linalg.norm(r2_0 - r1_0)# gives me one number v_0

inv_arel = 2/d - (v_0**2 / (G * (m1 +  m2)) )
a_rel = 1 / inv_arel

Time_period = 2 * np.pi* np.sqrt(a_rel**3 / (G *  (m1 + m2)))
T10 = Time_period * 10

#move all in the center
move_to_center_of_mass_frame(bodies)

for body in bodies: 
    body.record_state()

print('Int tot momentum:', total_momentum(bodies))
print('Int center of mass:', center_of_mass_position(bodies))

n_steps = 2000
h = T10 / n_steps

time_values = []
for steps in range (n_steps):
    vel_verlet(bodies, h, G)
    time_values.append((steps+1) * h )

time_values = np.array(time_values)

#change into arrays
positions1= np.array(body1.position_history)
positions2= np.array(body2.position_history)

fig1 = plt.figure()
ax = fig1.add_axes([0.12, 0.12, 0.80, 0.80])

#trajectories
ax.plot(positions1[:,0], positions1[:,1], label = 'Body1')
ax.plot(positions2[:,0], positions2[:,1], label = 'Body2')

# Initial positions
ax.plot(positions1[0, 0], positions1[0, 1],
        marker='o', color='blue', markersize=7,
        linestyle='none', label='Body 1 initial position')

ax.plot(positions2[0, 0], positions2[0, 1],
        marker='o', color='orange', markersize=7,
        linestyle='none', label='Body 2 initial position')

# Center of mass: it is at the origin in the CM frame
ax.plot(0, 0, marker='*', color='black',
        markersize=12, linestyle='none',
        label='Center of mass')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Comparable-Mass Binary Trajectories')
ax.set_aspect('equal', adjustable='box')
ax.grid()
ax.legend()

plt.show()
