

import numpy as np
from scipy.optimize import linprog





if __name__=="__main__":


    D = [1,0,0,0,0,0,0,0,0]
    A = [
        [-3.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0, 6.0, 8.0],
         [0, -1.0, -3.0, -2.0, -3.0, -4.0, -2.0, -3.0, -5.0 ]
         ]
    B = [0, -3.]

    ub = []
    lb = []
    result = linprog( D, A_ub=A, b_ub=B)

    print(f"optimization results: {result.fun} x vals {result.x}")

    print("done")
