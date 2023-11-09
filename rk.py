import numpy as np


def cds_magic(phi, phi_plus_one, phi_minus_one, domain, CONST_DICT: dict):
    U = CONST_DICT['U']
    GAMMA = CONST_DICT['GAMMA']
    RHO = CONST_DICT['RHO']

    DELTA_X = domain[1]

    # dphi_dt = ((-1 * (U)) * ( (phi_plus_one - phi_minus_one)) / (2 * DELTA_X) ) + ( ( GAMMA/RHO ) * ( (phi_plus_one + phi_minus_one - 2 * phi) / (DELTA_X**2) ) )
    #

    dphi_dt = -U * ((phi_plus_one - phi_minus_one) /(2*DELTA_X)) + (GAMMA/RHO) * ((phi_plus_one + phi_minus_one - 2 * phi)/(DELTA_X**2))
    return dphi_dt

def rk2(initial_guess: np.array, CONST_DICT, analytical_solution, domain, MAX_STEPS, DELTA_T, checkpoint, verbose=False):

    eps = 0.001
    solution = np.zeros(len(domain), dtype=np.double)

    history = []

    for t in range(0, int(MAX_STEPS)):
        if t == 0:
            solution = initial_guess.copy()
        if t == 1:
            solution[0] = CONST_DICT['PHI_LEFT']
            solution[-1] = CONST_DICT['PHI_RIGHT']
        if np.sum(np.abs(analytical_solution - solution)) <= eps:
            break

        #UPDATE
        prev_solution = solution.copy()
        Q_STAR = prev_solution.copy()
        for i in range(1, len(domain)-1):
            #CORRECTOR
            Q_STAR[i] = prev_solution[i] + ((DELTA_T/2) * cds_magic(
                phi=prev_solution[i],
                phi_plus_one=prev_solution[i+1],
                phi_minus_one=prev_solution[i-1],
                domain=domain,
                CONST_DICT=CONST_DICT
            ))

        for i in range(1, len(domain)-1):
            #PREDICTOR
            solution[i] = prev_solution[i] + (DELTA_T * cds_magic(
                phi=Q_STAR[i],
                phi_plus_one=Q_STAR[i + 1],
                phi_minus_one=Q_STAR[i - 1],
                domain=domain,
                CONST_DICT=CONST_DICT
            ))
        # if np.sum(np.abs(solution - prev_solution)) <= eps:
        #     history.append(solution)
        #     break
        # if verbose:
        #     print(solution)
        if t % checkpoint == 0:
            print(t, "Saving to history")
            # print(f"This is QSTAR{Q_STAR}")
            history.append(solution)














    return solution, history

def runge_kutta(initial_guess: np.array, order:int, analytical_solution, CONST_DICT:dict, DELTA_T, domain,  MAX_STEPS=10e6, checkpoint=1, verbose=False):
    solution = None


    if order == 2:
        print("USING RK2")
        solution, history = rk2(initial_guess=initial_guess.copy(), CONST_DICT=CONST_DICT, analytical_solution=analytical_solution, domain=domain, MAX_STEPS=MAX_STEPS, DELTA_T=DELTA_T, checkpoint=checkpoint, verbose=verbose)

    else:
        raise NotImplementedError("ONLY RK2 is implemented, set order = 2")




    return solution, history