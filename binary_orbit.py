import numpy as np  
from body import Body


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

Time_period = 2 * np.pi* (a_rel**3 / (G (m1 + m2)) )**(1/2)
T10 = Time_period * 10

a_i = 