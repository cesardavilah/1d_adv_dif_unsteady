import numpy as np
import matplotlib.pyplot as plt

def cds_magic(phi, phi_plus_one, phi_minus_one, domain, CONST_DICT: dict):
    U = CONST_DICT['U']
    GAMMA = CONST_DICT['GAMMA']
    RHO = CONST_DICT['RHO']

    DELTA_X = domain[1]

    # dphi_dt = ((-1 * (U)) * ( (phi_plus_one - phi_minus_one)) / (2 * DELTA_X) ) + ( ( GAMMA/RHO ) * ( (phi_plus_one + phi_minus_one - 2 * phi) / (DELTA_X**2) ) )
    #

    dphi_dt = -U * ((phi_plus_one - phi_minus_one) /(2*DELTA_X)) + (GAMMA/RHO) * ((phi_plus_one + phi_minus_one - 2 * phi)/(DELTA_X**2))
    return dphi_dt

def dphi_dt(solution, domain, CONST_DICT, verbose):
    '''
    Returns dphi/dt for all points in domain
    :param domain:
    :param verbose:
    :return:
    '''
    dphi_dt_domain = np.zeros(len(solution))
    for i in range(1, len(solution) - 1):
        dphi_dt_domain[i] = cds_magic(
                phi=solution[i],
                phi_plus_one=solution[i+1],
                phi_minus_one=solution[i-1],
                domain=domain,
                CONST_DICT=CONST_DICT
            )


    return dphi_dt_domain



def rk2_revision(initial_guess: np.array, CONST_DICT, analytical_solution, domain, MAX_STEPS, DELTA_T, checkpoint, verbose=False):


    solution = initial_guess.copy()
    history = []


    if verbose:
        plt.figure()
        print('DEBUGGING INSIDE @rk2_revision')

    for t in range(0, int(MAX_STEPS)):
        if t == 1:
            solution[-1] = CONST_DICT['PHI_RIGHT']
        # PREDICTOR
        Q_STAR = solution + (DELTA_T/2)*dphi_dt(solution=solution, domain=domain, CONST_DICT=CONST_DICT, verbose=verbose)
        # CORRECTOR
        solution = solution + DELTA_T * dphi_dt(solution=Q_STAR, domain=domain, CONST_DICT=CONST_DICT, verbose=verbose)

        if t % checkpoint == 0:
            print(t, "Saving to history")
            history.append(solution)

        if verbose: print(solution)
        if verbose: plt.plot(domain, solution)

    if verbose: plt.show()


    return solution, history

def runge_kutta(initial_guess: np.array, order:int, analytical_solution, CONST_DICT:dict, DELTA_T, domain,  MAX_STEPS=10e6, checkpoint=1, verbose=False):
    solution = None


    if order == 2:
        print("USING RK2")
        # solution, history = rk2(initial_guess=initial_guess.copy(), CONST_DICT=CONST_DICT, analytical_solution=analytical_solution, domain=domain, MAX_STEPS=MAX_STEPS, DELTA_T=DELTA_T, checkpoint=checkpoint, verbose=verbose)
        solution, history = rk2_revision(initial_guess=initial_guess.copy(), CONST_DICT=CONST_DICT, analytical_solution=analytical_solution, domain=domain, MAX_STEPS=MAX_STEPS, DELTA_T=DELTA_T, checkpoint=checkpoint, verbose=verbose)

    else:
        raise NotImplementedError("ONLY RK2 is implemented, set order = 2")




    return solution, history