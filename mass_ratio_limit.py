import numpy as np
import matplotlib.pyplot as plt
from body import Body
from simulation import (move_to_center_of_mass_frame, vel_verlet,
                        total_momentum, center_of_mass_position,
                        system_energy)

m_light = 1e-6
m_heavy = 1.0
d = 1.0
G = 1.0

r1_heavy = np.array([0.0, 0.0])
r2_light = np.array([d, 0.0])

v1_heavy = np.array([0.0, 0.8])
v2_light = np.array([0.0, 0.0])

body1 = Body(mass=m_light, position = r1_heavy, velocity = v1_heavy)
body2 = Body(mass=m_heavy, position = r2_light, velocity = v2_light)

bodies = [body1, body2]


print('Preliminary heavy velocity:', body1.velocity)
print('Preliminary light velocity:', body2.velocity)
#move all in the center
move_to_center_of_mass_frame(bodies)

print('CM-frame heavy velocity:', body1.velocity)
print('CM-frame light velocity:', body2.velocity)


for body in bodies: 
    body.record_state()


v_0 = np.linalg.norm(v2_light - v1_heavy)# gives me one number v_0
r_0 = np.linalg.norm(r2_light - r1_heavy)# gives me one number v_0

print(v_0)

inv_arel = 2/d - (v_0**2 / (G * (m_light +  m_heavy)) )
a_rel = 1 / inv_arel


Time_period = 2 * np.pi* np.sqrt(a_rel**3 / (G *  (m_light + m_heavy)))
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

fig1 = plt.figure()
ax = fig1.add_axes([0.12, 0.12, 0.80, 0.80])

#trajectories
ax.plot(positions1[:,0], positions1[:,1], label = 'm_heavy')
ax.plot(positions2[:,0], positions2[:,1], label = 'm_light')

# Initial positions
ax.plot(positions1[0, 0], positions1[0, 1],
        marker='o', color='blue', markersize=7,
        linestyle='none', label='m_heavy initial position')

ax.plot(positions2[0, 0], positions2[0, 1],
        marker='o', color='orange', markersize=7,
        linestyle='none', label='m_light initial position')

# # Center of mass: it is at the origin in the CM frame
# ax.plot(0, 0, marker='*', color='black',
#         markersize=12, linestyle='none',
#         label='Center of mass')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(' Elliptical orbit')
ax.set_aspect('equal', adjustable='box')
ax.grid()
ax.legend()

#zoom 

plt.show()

positions_heavy_ellipse = np.array(body1.position_history)
positions_light_ellipse = np.array(body2.position_history)
time_ellipse = time_values.copy()

#Energie plot
#calculate the Energie 
K_all, U_all, E_all, K_system, U_system, E_system = system_energy(bodies, G)


fig2, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# Kinetic energy
axes[0].plot(time_values, K_all[:, 0],
             label='m_heavy')

axes[0].plot(time_values, K_all[:, 1],
             label='m_light')

axes[0].plot(time_values, K_system,
             color='black', linestyle='--',
             label='System total')

axes[0].set_ylabel('Kinetic energy K')
axes[0].set_title('Energy History of Elliptical orbit')
axes[0].grid()
axes[0].legend()


# Potential energy
axes[1].plot(time_values, U_all[:, 0],
             label='m_heavy')

axes[1].plot(time_values, U_all[:, 1],
             label='m_light')

axes[1].plot(time_values, U_system,
             color='black', linestyle='--',
             label='System total')

axes[1].set_ylabel('Potential energy U')
axes[1].grid()
axes[1].legend()


# Total energy
axes[2].plot(time_values, E_all[:, 0],
             label='m_heavy')

axes[2].plot(time_values, E_all[:, 1],
             label='m_light')

axes[2].plot(time_values, E_system,
             color='black', linestyle='--',
             label='System total')

axes[2].set_xlabel('Time t')
axes[2].set_ylabel('Total energy E ')
axes[2].grid()
axes[2].legend()

plt.tight_layout()
plt.show()

#circular orbit

d = 1.0
r1_heavy = np.array([0.0, 0.0])
r2_light = np.array([d, 0.0])

v_c = np.sqrt(G * ( m_heavy + m_light) / d ) 
v1_heavy = np.array([0.0, 0.0])
v2_light = np.array([0.0, v_c])

body1 = Body(mass=m_light, position = r1_heavy, velocity = v1_heavy)
body2 = Body(mass=m_heavy, position = r2_light, velocity = v2_light)

bodies = [body1, body2]

print('Preliminary heavy velocity:', body1.velocity)
print('Preliminary light velocity:', body2.velocity)

move_to_center_of_mass_frame(bodies)

print('CM-frame heavy velocity:', body1.velocity)
print('CM-frame light velocity:', body2.velocity)

for body in bodies: 
    body.record_state()


v_0 = np.linalg.norm(v2_light - v1_heavy)# gives me one number v_0
r_0 = np.linalg.norm(r2_light - r1_heavy)# gives me one number v_0

print(v_0)

inv_arel = 2/d - (v_0**2 / (G * (m_light +  m_heavy)) )
a_rel = 1 / inv_arel


Time_period = (2 * np.pi* d) / v_c
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

fig3 = plt.figure()
ax = fig3.add_axes([0.12, 0.12, 0.80, 0.80])

#trajectories
ax.plot(positions1[:,0], positions1[:,1], label = 'm_heavy')
ax.plot(positions2[:,0], positions2[:,1], label = 'm_light')

# Initial positions
ax.plot(positions1[0, 0], positions1[0, 1],
        marker='o', color='blue', markersize=7,
        linestyle='none', label='m_heavy initial position')

ax.plot(positions2[0, 0], positions2[0, 1],
        marker='o', color='orange', markersize=7,
        linestyle='none', label='m_light initial position')

# # Center of mass: it is at the origin in the CM frame
# ax.plot(0, 0, marker='*', color='black',
#         markersize=12, linestyle='none',
#         label='Center of mass')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(' Circular orbit')
ax.set_aspect('equal', adjustable='box')
ax.grid()
ax.legend()
plt.show()

positions_heavy_circle = np.array(body1.position_history)
positions_light_circle = np.array(body2.position_history)
time_circle = time_values.copy()

#Energie plot
#calculate the Energie 
K_all, U_all, E_all, K_system, U_system, E_system = system_energy(bodies, G)


fig4, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# Kinetic energy
axes[0].plot(time_values, K_all[:, 0],
             label='m_heavy')

axes[0].plot(time_values, K_all[:, 1],
             label='m_light')

axes[0].plot(time_values, K_system,
             color='black', linestyle='--',
             label='System total')

axes[0].set_ylabel('Kinetic energy K')
axes[0].set_title('Energy History of Circular orbit')
axes[0].grid()
axes[0].legend()


# Potential energy
axes[1].plot(time_values, U_all[:, 0],
             label='m_heavy')

axes[1].plot(time_values, U_all[:, 1],
             label='m_light')

axes[1].plot(time_values, U_system,
             color='black', linestyle='--',
             label='System total')

axes[1].set_ylabel('Potential energy U ')
axes[1].grid()
axes[1].legend()


# Total energy
axes[2].plot(time_values, E_all[:, 0],
             label='m_heavy')

axes[2].plot(time_values, E_all[:, 1],
             label='m_light')

axes[2].plot(time_values, E_system,
             color='black', linestyle='--',
             label='System total')

axes[2].set_xlabel('Time t')
axes[2].set_ylabel('Total energy E ')
axes[2].grid()
axes[2].legend()

plt.tight_layout()
plt.show()

#maximum displacment from the center of mass in each case: 

max_heavy_ellipse = np.max(np.linalg.norm(positions_heavy_ellipse, axis=1))
max_heavy_circle = np.max(np.linalg.norm(positions_heavy_circle, axis=1))

print('Maximum heavy-body displacement, ellipse:', max_heavy_ellipse)
print('Maximum heavy-body displacement, circle:', max_heavy_circle)


#from assignment 6 

# Fixed-center elliptical orbit
# The relative velocity is v_light - v_heavy = (0, -0.8)
r_fixed = np.array([1.0, 0.0])
v_fixed = np.array([0.0, -0.8])

r_magnitude = np.linalg.norm(r_fixed)
acc_fixed = -G * m_heavy * r_fixed / r_magnitude**3

v0_fixed = np.linalg.norm(v_fixed)

semi_major_fixed = 1 / (2 / d - v0_fixed**2 / (G * m_heavy))
T_fixed_ellipse = 2 * np.pi * np.sqrt(
    semi_major_fixed**3 / (G * m_heavy)
)

fixed_steps = 2000
h_fixed = 10 * T_fixed_ellipse / fixed_steps

fixed_ellipse = [r_fixed]

for i in range(fixed_steps):

    r_next = r_fixed + v_fixed * h_fixed + 0.5 * acc_fixed * h_fixed**2

    r_next_magnitude = np.linalg.norm(r_next)
    acc_next = -G * m_heavy * r_next / r_next_magnitude**3

    v_next = v_fixed + 0.5 * (acc_fixed + acc_next) * h_fixed

    r_fixed = r_next
    v_fixed = v_next
    acc_fixed = acc_next

    fixed_ellipse.append(r_fixed)

fixed_ellipse = np.array(fixed_ellipse)


# Fixed-center circular orbit
r_fixed = np.array([1.0, 0.0])

v_c_fixed = np.sqrt(G * m_heavy / d)
v_fixed = np.array([0.0, v_c_fixed])

r_magnitude = np.linalg.norm(r_fixed)
acc_fixed = -G * m_heavy * r_fixed / r_magnitude**3

T_fixed_circle = 2 * np.pi * d / v_c_fixed
h_fixed = 10 * T_fixed_circle / fixed_steps

fixed_circle = [r_fixed]

for i in range(fixed_steps):

    r_next = r_fixed + v_fixed * h_fixed + 0.5 * acc_fixed * h_fixed**2

    r_next_magnitude = np.linalg.norm(r_next)
    acc_next = -G * m_heavy * r_next / r_next_magnitude**3

    v_next = v_fixed + 0.5 * (acc_fixed + acc_next) * h_fixed

    r_fixed = r_next
    v_fixed = v_next
    acc_fixed = acc_next

    fixed_circle.append(r_fixed)

fixed_circle = np.array(fixed_circle)
#comparison Fixd center orbit

relative_ellipse = positions_light_ellipse - positions_heavy_ellipse
relative_circle = positions_light_circle - positions_heavy_circle




fig5 = plt.figure()
ax5 = fig5.add_axes([0.12, 0.12, 0.80, 0.80])

ax5.plot(relative_ellipse[:, 0], relative_ellipse[:, 1],
         label='Two-body relative elliptical orbit')

ax5.plot(fixed_ellipse[:, 0], fixed_ellipse[:, 1],
         linestyle='--', label='Fixed-center elliptical orbit')

ax5.set_xlabel('x')
ax5.set_ylabel('y')
ax5.set_title('Elliptical Orbit: Two-Body vs. Fixed-Center')
ax5.set_aspect('equal', adjustable='box')
ax5.grid()
ax5.legend()

plt.show()

fig6 = plt.figure()
ax6 = fig6.add_axes([0.12, 0.12, 0.80, 0.80])

ax6.plot(relative_circle[:, 0], relative_circle[:, 1],
         label='Two-body relative circular orbit')

ax6.plot(fixed_circle[:, 0], fixed_circle[:, 1],
         linestyle='--', label='Fixed-center circular orbit')

ax6.set_xlabel('x')
ax6.set_ylabel('y')
ax6.set_title('Circular Orbit: Two-Body vs. Fixed-Center')
ax6.set_aspect('equal', adjustable='box')
ax6.grid()
ax6.legend()

plt.show()

difference_ellipse = np.linalg.norm(
    relative_ellipse - fixed_ellipse, axis=1
)

difference_circle = np.linalg.norm(
    relative_circle - fixed_circle, axis=1
)

print('Maximum ellipse-coordinate difference:',
      np.max(difference_ellipse))

print('Maximum circle-coordinate difference:',
      np.max(difference_circle))