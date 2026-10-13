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


#move all in the center
move_to_center_of_mass_frame(bodies)


initial_momentum = total_momentum(bodies)
print('Initial total momentum:', initial_momentum)

for body in bodies: 
    body.record_state()


v_0 = np.linalg.norm(v2 - v1)# gives me one number v_0
r_0 = np.linalg.norm(r2 - r1)# gives me one number v_0

inv_arel = 2/d - (v_0**2 / (G * (m1 +  m2)))
a_rel = 1 / inv_arel


Time_period = 2 * np.pi* np.sqrt(a_rel**3 / (G *  (m1 + m2)))
T10 = Time_period * 10
n_steps = 10000
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
        marker='o', color='green', markersize=7,
        linestyle='none', label='m3 initial position')


ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Three body ejection')
ax.set_aspect('equal', adjustable='datalim')
ax.grid()
ax.legend()

#zoom Plot

positions_m1_ellipse = np.array(body1.position_history)
positions_m2_ellipse = np.array(body2.position_history)
positions_m3_ellipse = np.array(body3.position_history)

time_ellipse = time_values.copy()

ax_inset = ax.inset_axes([0.58, 0.10, 0.35, 0.35])

ax_inset.plot(positions_m1_ellipse[:, 0],
              positions_m1_ellipse[:, 1], 
              color = 'blue')

ax_inset.plot(positions_m2_ellipse[:, 0],
              positions_m2_ellipse[:, 1], 
              color = 'orange')

ax_inset.plot(positions_m3_ellipse[:, 0],
              positions_m3_ellipse[:, 1], 
              color = 'green')

max_heavy = np.max(
    np.linalg.norm(positions_m3_ellipse, axis=1)
)

ax_inset.set_xlim(-0.2 * max_heavy, 0.2* max_heavy)
ax_inset.set_ylim(-0.2 * max_heavy, 0.2* max_heavy)
ax_inset.set_title('closeup', fontsize=8)
ax_inset.grid()
ax_inset.set_aspect('equal')

plt. show()

#Energie plot
#calculate the Energie 
K_all, U_all, E_all, K_system, U_system, E_system = system_energy(bodies, G)


fig2, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# Kinetic energy
axes[0].plot(time_values, K_all[:, 0],
             label='m1')

axes[0].plot(time_values, K_all[:, 1],
             label='m2')

axes[0].plot(time_values, K_all[:, 2],
             label='m3')


axes[0].plot(time_values, K_system,
             color='black', linestyle='--',
             label='System total')

axes[0].set_ylabel('Kinetic energy K')
axes[0].set_title('Energy History of three body orbits')
axes[0].grid()
axes[0].legend()


# Potential energy
axes[1].plot(time_values, U_all[:, 0],
             label='m1')

axes[1].plot(time_values, U_all[:, 1],
             label='m2')

axes[1].plot(time_values, U_all[:, 2],
             label='m3')

axes[1].plot(time_values, U_system,
             color='black', linestyle='--',
             label='System total')

axes[1].set_ylabel('Potential energy U')
axes[1].grid()
axes[1].legend()


# Total energy
axes[2].plot(time_values, E_all[:, 0],
             label='m1')

axes[2].plot(time_values, E_all[:, 1],
             label='m2')

axes[2].plot(time_values, E_all[:, 2],
             label='m3')

axes[2].plot(time_values, E_system,
             color='black', linestyle='--',
             label='System total')

axes[2].set_xlabel('Time t')
axes[2].set_ylabel('Total energy E ')
axes[2].grid()
axes[2].legend()

plt.tight_layout()
plt.show()

# pairwise distances

position_12= positions_m1_ellipse - positions_m2_ellipse
position_13= positions_m1_ellipse - positions_m3_ellipse
position_23= positions_m2_ellipse - positions_m3_ellipse

#print
fig1 = plt.figure()
ax = fig1.add_axes([0.12, 0.12, 0.80, 0.80])

#trajectories
ax.plot(position_12[:,0], position_12[:,1], label = 'r12')
ax.plot(position_13[:,0], position_13[:,1], label = 'r13')
ax.plot(position_23[:,0], position_23[:,1], label = 'r23')

# Initial positions
ax.plot(position_12[0, 0], position_12[0, 1],
        marker='o', color='blue', markersize=7,
        linestyle='none', label='r_12 initial position')

ax.plot(position_13[0, 0], position_13[0, 1],
        marker='o', color='orange', markersize=7,
        linestyle='none', label='r_13 initial position')

ax.plot(position_23[0, 0], position_23[0, 1],
        marker='o', color='green', markersize=7,
        linestyle='none', label='r_23 initial position')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Pairwise distances')
ax.set_aspect('equal', adjustable='datalim')
ax.grid()
ax.legend()

#closeup

ax_inset = ax.inset_axes([0.58, 0.10, 0.35, 0.35])

ax_inset.plot(positions_m1_ellipse[:, 0],
              positions_m1_ellipse[:, 1], 
              color = 'blue')

ax_inset.plot(positions_m2_ellipse[:, 0],
              positions_m2_ellipse[:, 1], 
              color = 'orange')

ax_inset.plot(positions_m3_ellipse[:, 0],
              positions_m3_ellipse[:, 1], 
              color = 'green')

max_heavy = np.max(
    np.linalg.norm(positions_m3_ellipse, axis=1)
)

ax_inset.set_xlim(-0.2 * max_heavy, 0.2* max_heavy)
ax_inset.set_ylim(-0.2 * max_heavy, 0.2* max_heavy)
ax_inset.set_title('closeup', fontsize=8)
ax_inset.grid()
ax_inset.set_aspect('equal')

plt. show()

#calculating the min pariwise distacnc
magnitudes_12 = np.linalg.norm(position_12, axis=1)
magnitudes_13 = np.linalg.norm(position_13, axis=1)
magnitudes_23 = np.linalg.norm(position_23, axis=1)

smallest_magnitude12 = np.min(magnitudes_12)
smallest_magnitude13 = np.min(magnitudes_13)
smallest_magnitude23 = np.min(magnitudes_23)

print(f'minimum pariwise distance between Body1 and Body2:{smallest_magnitude12}')
print(f'minimum pariwise distance between Body1 and Body3:{smallest_magnitude13}')
print(f'minimum pariwise distance between Body2 and Body3:{smallest_magnitude23}')

minimum_pairwise_distance = min(
    smallest_magnitude12,
    smallest_magnitude13,
    smallest_magnitude23
)
print()
print(f'Overall minimum pairwise distance: {minimum_pairwise_distance:.6e}')

#Final bound

μ_12 = m1*m2 / (m1+m2)

velocity_m1 = np.array(body1.velocity_history[-1])
velocity_m2 = np.array(body2.velocity_history[-1])

diff_v1v2 = velocity_m1 - velocity_m2
magnitudes_diffv1v2 = np.linalg.norm(diff_v1v2)

r_12 = np.linalg.norm(body1.position_history[-1] - body2.position_history[-1])# gives me one number v_0

E_12 = 1/2 * μ_12 * (magnitudes_diffv1v2)**2 - (G*m1*m2/r_12)

r_13 = np.linalg.norm(
    body1.position_history[-1] - body3.position_history[-1]
)
v_13 = np.linalg.norm(
    body1.velocity_history[-1] - body3.velocity_history[-1]
)
mu_13 = (m1 * m3) / (m1 + m3)
E_13 = 0.5 * mu_13 * v_13**2 - (G * m1 * m3 / r_13)


r_23 = np.linalg.norm(
    body2.position_history[-1] - body3.position_history[-1]
)
v_23 = np.linalg.norm(
    body2.velocity_history[-1] - body3.velocity_history[-1]
)
mu_23 = (m2 * m3) / (m2 + m3)
E_23 = 0.5 * mu_23 * v_23**2 - (G * m2 * m3 / r_23)

print()
print(f'Final pair energy E_12: {E_12}')
print(f'Final pair energy E_13: {E_13}')
print(f'Final pair energy E_23: {E_23}')

# ramaining body

# Center of mass of the final binary 
binary_cm = ( m1 * positions1 + m3 * positions2)/ (m1 + m3)

# Distance of the remaining body from the binary CM
r2_from_binary = positions2 - binary_cm

distance_to_binary = np.linalg.norm(r2_from_binary, axis=1)

#plot 
fig4 = plt.figure()
ax4 = fig4.add_axes([0.12, 0.12, 0.80, 0.80])

ax4.plot(time_values, distance_to_binary)

ax4.set_xlabel('Time t')
ax4.set_ylabel('Distance from binary center of mass')
ax4.set_title('Remaining Body Moving Away from Final Binary')
ax4.grid()

plt.show()

#diviation Plot


# Initial total momentum in the center-of-mass fram
#did it after moving the body to the center bofor

# Center-of-mass position at every stored time
cm_positions = (m1 * positions1 + m2 * positions2 + m3* positions3) / (m1 + m2 + m3)

# Deviation from the initial center-of-mass position
cm_deviation = np.linalg.norm(cm_positions - cm_positions[0], axis=1)


print('Maximum center-of-mass deviation:',
      np.max(cm_deviation))

# Plot center-of-mass deviation
fig5 = plt.figure()
ax5 = fig5.add_axes([0.12, 0.12, 0.80, 0.80])

ax5.plot(time_values, cm_deviation)
ax5.set_xlabel('Time t')
ax5.set_ylabel('Center-of-mass deviation')
ax5.set_title('Center-of-Mass Deviation')
ax5.grid()

plt.show()