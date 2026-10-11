import numpy as np
from body import Body
import matplotlib.pyplot as plt



# two body acceleration
def pair_acc (body_i: Body,
         body_j: Body, 
         G: float,) -> list[np.ndarray]:

    displacement = body_j.position - body_i.position

    #distance of two dodies
    #distance_ij = (displacement**2)**(1/2)

    distance_ij = np.linalg.norm(displacement)# gives me one number r_ij
    F_ij = G * body_i.mass * body_j.mass * (displacement/distance_ij**3)

    #acc from one body to another
    pair_acc = F_ij/body_i.mass

    return pair_acc

#multiebody acceleration
def all_acc(bodies: list[Body],
            G: float)->np.ndarray:

    #list of all acceleration vectors acting on body_i
    accelerations = []

    for body_i in bodies: 

        total_acceleration = np.zeros_like(body_i.position)

        for body_j in bodies: 
            # prevents bodies to act on themself
            if body_i is not body_j: 
                #creats single acc vector for the foce of two bodies
                total_acceleration += pair_acc(body_i, body_j, G)


        accelerations.append(total_acceleration)

    return accelerations

#velocity

def vel_verlet(bodies: list[Body], 
               h: float, 
               G: float)->None:

    #all acceleration at timt t
    accelerations_old = all_acc(bodies, G)

    new_positions = []

    #
    for i in range(len(bodies)):


        body = bodies[i]

        position_new = body.position + body.velocity * h + 0.5 * accelerations_old[i] * h**2

        new_positions.append(position_new)

    #apply all new postions simultaneously
    for i in range(len(bodies)):
        bodies[i].position = new_positions[i]

    #calculation of all accelerations for the new postions
    accelerations_new = all_acc(bodies, G)

    #calculation of velocity 
    for i in range(len(bodies)):

        bodies[i].velocity = bodies[i].velocity + 0.5 * (accelerations_old[i] + accelerations_new[i]) * h

    for body in bodies: 
        body.record_state()


def system_energy(bodies: list[Body],
                  G: float):

    
    # Number of stored time values
    N = len(bodies[0].position_history)

    #List for Energie over time 
    K_all = []
    U_all = []
    E_all = []

    for iter in range (N):
        #Kinetic Energie
        K_values = []
        for i in range(len(bodies)):
            v_i = np.linalg.norm(bodies[i].velocity_history[iter])# gives me one number v_i
            K = 1/2 * bodies[i].mass * v_i**2 
            K_values.append(K)
        K_sum = sum(K_values)
        


        #Potential energy
        # Potential energy of every body
        U_values = np.zeros(len(bodies))
        #U_pot = []
        for i in range(len(bodies)-1):
            for j in range(i+1, (len(bodies))):

                displacement = bodies[i].position_history[iter] - bodies[j].position_history[iter]
                distance_ij = np.linalg.norm(displacement)# gives me one number r_ij

                U = -(G * bodies[i].mass * bodies[j].mass) / distance_ij
                #U_pot.append(U)
                 # Half of the pair energy belongs to each body
                U_values[i] += 0.5 * U
                U_values[j] += 0.5 * U

        K_values = np.array(K_values)

        # Total energy of each individual body
        E_values = K_values + U_values

        # Store individual body energies at this time
        K_all.append(K_values)
        U_all.append(U_values)
        E_all.append(E_values)

    # Convert energy histories to NumPy arrays
    K_all = np.array(K_all)
    U_all = np.array(U_all)
    E_all = np.array(E_all)

    # Sum over all bodies at every time value
    K_system = np.sum(K_all, axis=1)
    U_system = np.sum(U_all, axis=1)
    E_system = np.sum(E_all, axis=1)

    return K_all, U_all, E_all, K_system, U_system, E_system

#center of Mass and Momentum
def total_momentum(bodies: list[Body]):

    p_values = []
    for i in range(len(bodies)):
        p = bodies[i].mass * bodies[i].velocity
        p_values.append(p)

    return np.sum(p_values, axis=0)

def center_of_mass_position(bodies: list[Body]):

    weighted_positions = []
    masses = []

    for i in range(len(bodies)): 
        weighted_position = bodies[i].mass * bodies[i].position
        weighted_positions.append(weighted_position)

        masses.append(bodies[i].mass)
    return np.sum(weighted_positions, axis=0)/ np.sum(masses)

def center_of_mass_velocity(bodies: list[Body]):

    weighted_velocitys = []
    masses = []

    for i in range(len(bodies)):
        weighted_velocity = bodies[i].mass * bodies[i].velocity
        weighted_velocitys.append(weighted_velocity)

        masses.append(bodies[i].mass)
    return np.sum(weighted_velocitys, axis=0)/ np.sum(masses)

def move_to_center_of_mass_frame(bodies: list[Body]):

    position_cm = center_of_mass_position(bodies)
    velocity_cm = center_of_mass_velocity(bodies)

    for i in range(len(bodies)):
        #changes the position and velocity direct, so no return necassary
        bodies[i].position = bodies[i].position - position_cm
        
        bodies[i].velocity = bodies[i].velocity - velocity_cm

#guess the period , take the index of the max_kin_values and use the associated time value
#calculate the time between two max = period

def measure_period_from_energy(kinetic_energy,
                 time_values):
    peak_indices = []

    for i in range(len(kinetic_energy)):
        if (kinetic_energy[i] > kinetic_energy[i-1]
        and kinetic_energy[i] > kinetic_energy[i+1]):
            peak_indices.append(i)

    peak_times = time_values[peak_indices]
    periods = np.diff(peak_times)

    return np.mean(periods)