import numpy as np
import matplotlib.pyplot as plt


import cds
import rk



import sec



if __name__ == '__main__':

    #PATHS
    path_to_save_figs = sec.path_to_figures

    #CONSTANTS
    U = 4.0
    L = 5.0
    RHO = 900.0
    GAMMA = 650.0

    PHI_LEFT_INITIAL = 3.0
    PHI_RIGHT_INITIAL = 3.0

    PHI_LEFT = 3.0
    PHI_RIGHT = 1.0

    #PACKING CONSTANTS
    CONST_DICT = {
        'U': U,
        'L': L,
        'RHO': RHO,
        'GAMMA': GAMMA,

        'PHI_LEFT_INITIAL': PHI_LEFT_INITIAL,
        'PHI_RIGHT_INITIAL': PHI_RIGHT_INITIAL,

        'PHI_LEFT': PHI_LEFT,
        'PHI_RIGHT': PHI_RIGHT

    }

    EX_RATIO = 1 #CONSTANT SPACING
    N = 40
    DELTA_T = .01
    STEPS = 10000






    #discretize
    domain, delta_x = cds.discretize_domain(len_domain=L, expansion_ratio=EX_RATIO, number_of_elements=N, verbose=False)

    #initialize

    initial_guess = cds.initialize_domain(domain=domain, CONST_DICT=CONST_DICT, interpolation_method="linear", verbose=False)

    #analytical solution
    analytical_solution = cds.exact_solution(domain=domain, CONST_DICT=CONST_DICT)

    #use RK2
    checkpoint = 1
    rk_2_solution, history = rk.runge_kutta(initial_guess=initial_guess, order=2, analytical_solution=analytical_solution, CONST_DICT=CONST_DICT, DELTA_T=DELTA_T, domain=domain, MAX_STEPS=STEPS, checkpoint=checkpoint, verbose=True)

    # plt.plot(domain, rk_2_solution)
    # plt.show()
    plt.figure()
    for h in history:
        plt.plot(domain, h)
        # print(h)

    plt.show()


    # plt.show()




    #present solution