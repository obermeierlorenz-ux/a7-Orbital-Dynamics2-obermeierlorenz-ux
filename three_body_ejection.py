import numpy as np
import matplotlib.pyplot as plt
from body import Body
from simulation import (move_to_center_of_mass_frame, vel_verlet,
                        total_momentum, center_of_mass_position,
                        system_energy)

m1 = 1.0
m2 = 1.0
m3 = 1.0
d = 1.0
G = 1.0

r1 = np.array([-0.5, 0.0])
r2 = np.array([0.5, 0.0])
r3 = np.array([0.0, 5.0])

v1 = np.array([0.0, -1/np.sqrt(2)])
v2 = np.array([0.0, 1/np.sqrt(2)])
v3 = np.array([0.0, -0.3])

body1 = Body(mass=m1, position = r1, velocity = v1)
body2 = Body(mass=m2, position = r2, velocity = v2)
body3 = Body(mass=m3, position = r3, velocity = v3)

bodies = [body1, body2, body3]


print('Preliminary heavy velocity:', body1.velocity)
print('Preliminary light velocity:', body2.velocity)
#move all in the center
move_to_center_of_mass_frame(bodies)

print('CM-frame heavy velocity:', body1.velocity)
print('CM-frame light velocity:', body2.velocity)


for body in bodies: 
    body.record_state()


v_0 = np.linalg.norm(v2 - v1)# gives me one number v_0
r_0 = np.linalg.norm(r2 - r1)# gives me one number v_0

inv_arel = 2/d - (v_0**2 / (G * (m1 +  m2)))
a_rel = 1 / inv_arel


Time_period = 2 * np.pi* np.sqrt(a_rel**3 / (G *  (m1 + m2)))
T10 = Time_period * 10
n_steps = 2000
h = T10 / n_steps

time_values = [0.0]
for steps in range (n_steps):
    vel_verlet(bodies, h, G)
    time_values.append((steps+1) * h )

time_values = np.array(time_values)

#change into arrays
positions1= np.array(body1.position_history)
positions2= np.array(body2.position_history)
positions3= np.array(body3.position_history)


#print
fig1 = plt.figure()
ax = fig1.add_axes([0.12, 0.12, 0.80, 0.80])

#trajectories
ax.plot(positions1[:,0], positions1[:,1], label = 'm1')
ax.plot(positions2[:,0], positions2[:,1], label = 'm2')
ax.plot(positions3[:,0], positions3[:,1], label = 'm3')

# Initial positions
ax.plot(positions1[0, 0], positions1[0, 1],
        marker='o', color='blue', markersize=7,
        linestyle='none', label='m1 initial position')

ax.plot(positions2[0, 0], positions2[0, 1],
        marker='o', color='orange', markersize=7,
        linestyle='none', label='m2 initial position')

ax.plot(positions3[0, 0], positions3[0, 1],
        marker='o', color='orange', markersize=7,
        linestyle='none', label='m3 initial position')

# # Center of mass: it is at the origin in the CM frame
# ax.plot(0, 0, marker='*', color='black',
#         markersize=12, linestyle='none',
#         label='Center of mass')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Three body ejection')
ax.set_aspect('equal', adjustable='box')
ax.grid()
ax.legend()

#zoom Plot

positions_m1_ellipse = np.array(body1.position_history)
positions_m2_ellipse = np.array(body2.position_history)
positions_m3_ellipse = np.array(body3.position_history)
time_ellipse = time_values.copy()

ax_inset = ax.inset_axes([0.58, 0.10, 0.35, 0.35])

ax_inset.plot(positions_m1_ellipse[:, 0],
              positions_m1_ellipse[:, 1])

max_heavy = np.max(
    np.linalg.norm(positions_m1_ellipse, axis=1)
)

ax_inset.set_xlim(-1.1 * max_heavy, 1.1 * max_heavy)
ax_inset.set_ylim(-1.1 * max_heavy, 1.1 * max_heavy)
ax_inset.set_title('m1', fontsize=8)
ax_inset.grid()
ax_inset.set_aspect('equal')

plt. show()
