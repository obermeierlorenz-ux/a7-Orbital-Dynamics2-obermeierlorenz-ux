import numpy as np  
import matplotlib.pyplot as plt
from body import Body
from simulation import (move_to_center_of_mass_frame, vel_verlet,
                        total_momentum, center_of_mass_position,
                        system_energy,  measure_period_from_energy)


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

#Energie plot
#calculate the Energie 
K_all, U_all, E_all, K_system, U_system, E_system = system_energy(bodies, G)


fig2, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# Kinetic energy
axes[0].plot(time_values, K_all[:, 0],
             label='Body 1')

axes[0].plot(time_values, K_all[:, 1],
             label='Body 2')

axes[0].plot(time_values, K_system,
             color='black', linestyle='--',
             label='System total')

axes[0].set_ylabel('Kinetic energy K')
axes[0].set_title('Energy History of the Binary System')
axes[0].grid()
axes[0].legend()


# Potential energy
axes[1].plot(time_values, U_all[:, 0],
             label='Body 1')

axes[1].plot(time_values, U_all[:, 1],
             label='Body 2')

axes[1].plot(time_values, U_system,
             color='black', linestyle='--',
             label='System total')

axes[1].set_ylabel('Potential energy U')
axes[1].grid()
axes[1].legend()


# Total energy
axes[2].plot(time_values, E_all[:, 0],
             label='Body 1')

axes[2].plot(time_values, E_all[:, 1],
             label='Body 2')

axes[2].plot(time_values, E_system,
             color='black', linestyle='--',
             label='System total')

axes[2].set_xlabel('Time t')
axes[2].set_ylabel('Total energy E')
axes[2].grid()
axes[2].legend()

plt.tight_layout()
plt.show()

#measure period
#guess the period , take the index of the max_kin_values and use the associated time value
#calculate the time between two max = period

period_body1 = measure_period_from_energy(K_all[:,0], time_values)
period_body2 = measure_period_from_energy(K_all[:,1], time_values)

#comparison
period_diff = period_body2 - period_body1
print(f"period body1:{period_body1}")
print(f"period body1:{period_body2}")

print()
print(f"period difference between the two bodies:{period_diff}")
print()
comp = Time_period-period_body1
print (f'expected - measured : {comp}')

#periapsis apoapsis

r_body1 = np.linalg.norm(positions1, axis=1)
r_body2 = np.linalg.norm(positions2, axis=1)

r1_peri = np.min(r_body1)
r1_apo = np.max(r_body1)

r2_peri = np.min(r_body2)
r2_apo = np.max(r_body2)

a1 = 0.5 * (r1_apo + r1_peri)
a2 = 0.5 * (r2_apo + r2_peri)

#eccentricities
e1 = (r1_apo - r1_peri) / (r1_apo + r1_peri)
e2 = (r2_apo - r2_peri) / (r2_apo + r2_peri)

#comparison
print()
print('Body 1 periapsis:', r1_peri)
print('Body 1 apoapsis:', r1_apo)
print('Body 1 a:', a1)
print('Body 1 eccentricity:', e1)
print()
print('Body 2 periapsis:', r2_peri)
print('Body 2 apoapsis:', r2_apo)
print('Body 2 a:', a2)
print('Body 2 eccentricity:', e2)
print()
print('a1 / a2:', a1 / a2)
print('m2 / m1:', m2 / m1)



#last part of ex. 1
# Initial total momentum in the center-of-mass frame
initial_momentum = total_momentum(bodies)
print('Initial total momentum:', initial_momentum)

# Center-of-mass position at every stored time
cm_positions = (m1 * positions1 + m2 * positions2) / (m1 + m2)

# Deviation from the initial center-of-mass position
cm_deviation = np.linalg.norm(cm_positions - cm_positions[0], axis=1)

# Plot center-of-mass deviation
fig3 = plt.figure()
ax3 = fig3.add_axes([0.12, 0.12, 0.80, 0.80])

ax3.plot(time_values, cm_deviation)
ax3.set_xlabel('Time t')
ax3.set_ylabel('Center-of-mass deviation')
ax3.set_title('Center-of-Mass Deviation')
ax3.grid()

plt.show()



