"""
SEIR Model
"""

import numpy as np





methods = {
    'Euler': [[1]],
    'RK4': {'c': np.array([0,0.5,0.5,1]),
            'b': np.array([1/6,1/3,1/3,1/6]),
            'a': 'test'
            }
}

def solver(methods,f,y,t,h):
    """
    Explit solver to solve ordinary differential equations using explicit Runge-Kutta

    """

    k = []

    nbr_k = len(methods)


    for k_i in range(nbr_k):
        print(k_i)
        
        
