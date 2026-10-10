import numpy as np

class Body:
    def __init__(self, mass: float,
                 position: np.ndarray,
                 velocity: np.ndarray)->None:

        #saves the diverent parameter for the object
        self.mass = mass
        self.position = np.array(position, dtype = float)
        self.velocity = np.array(velocity, dtype = float)

        #history
        self.position_history = []
        self.velocity_history = []

    def record_state(self)-> None:

        self.position_history.append(self.position.copy())
        self.velocity_history.append(self.velocity.copy())
