import numpy as np
from body import Body


# two body acceleration
def pair_acc (body_i: Body,
         body_j: Body, 
         G: float,) -> list[np.ndarray]:

    displacement = body_i.position - body_j.position

    #distance of two dodies
    #distance_ij = (displacement**2)**(1/2)

    distance_ij = np.linalg.norm(displacement)# gives me one number r_ij
    F_ij = G * body_i.mass * body_j.mass * (displacement/distance_ij**3)

    #acc from one body to another
    pair_acc = F_ij/body_i.mass

    return pair_acc

#multiebody acceleration
def all_acc(bodies: list[Body],
            G: float)->list[np.ndarray]:

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


