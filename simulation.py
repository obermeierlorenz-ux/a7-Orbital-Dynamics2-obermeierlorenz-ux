import numpy as np
from body import Body


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
               h, float, 
               G: float)->None:

    #all acceleration at timt t
    accelerations_old = all_acc(bodies, G)

    new_positions = []

    #
    for i in range(len(bodies)):


        body = bodies[i]

        postion_new = body.position + body.velocity * h + 0.5 * accelerations_old[i] * h**2

    new_positions.append(postion_new)

    #apply all new postions simultaneously
    for i in range(len(bodies)):
        bodies[i].postion = new_positions[i]

    #calculation of all accelerations for the new postions
    accelerations_new = all_acc(bodies, G)

    #calculation of velocity 
    for i in range(len(bodies)):

        bodies[i].velocity = bodies[i].velocity + 0.5 * (accelerations_old[i] + accelerations_new[i]) * h

    for body in bodies: 
        body.record_state()