import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

import math



def exact_solution(domain, CONST_DICT: dict):
    phi_left = CONST_DICT['PHI_LEFT']
    phi_right = CONST_DICT['PHI_RIGHT']
    u = CONST_DICT['U']
    L = CONST_DICT['L']
    rho = CONST_DICT['RHO']
    gamma = CONST_DICT['GAMMA']



    pe = (rho * u * L) / (gamma)
    phi = np.zeros(len(domain))
    for i in range(0, len(domain)):
        phi[i] = ( phi_left + ((phi_right - phi_left) * ((math.exp(domain[i]*(pe/L)) - 1)/((math.exp(pe)) - 1))))

    return phi

def initialize_domain(domain: np.array, CONST_DICT: dict, interpolation_method: str, verbose=False):
    '''

    :param domain:
    :param PHI_LEFT:
    :param PHI_RIGHT:
    :param interpolation_method: How to fill the rest of the values in-between boundaries
    :return:
    '''
    # dom = np.zeros(len(domain))


    PHI_LEFT = CONST_DICT["PHI_LEFT_INITIAL"]
    PHI_RIGHT = CONST_DICT["PHI_RIGHT_INITIAL"]

    x = [domain[0], domain[-1]]
    y = [PHI_LEFT, PHI_RIGHT]



    if interpolation_method == 'linear':
        dom = np.interp(domain, x, y)


    if verbose:
        plt.plot(domain, dom)
        plt.show()





    return dom

def initial_deltax(len_domain: float, expansion_ratio: float, number_of_elements: int):
    '''
    :param len_domain: The lenght of the domain in meters
    :param expansion_ratio: The expansion ratio re as defined in lecture 9. For re>1 more elements on left. For re<1 more elements on right.
    :param number_of_elements: The number of elements desired, aka N
    :return: Initial delta_x as a value. This can be used to calculate the rest of the points in the domain.
    '''

    if expansion_ratio == 1:
        deltax = len_domain/number_of_elements
    else:
        deltax = len_domain * ((expansion_ratio - 1)/((expansion_ratio**number_of_elements) - 1))


    return deltax

def discretize_domain(len_domain: float, expansion_ratio: float, number_of_elements: int, verbose=False):
    '''
    :param len_domain:
    :param expansion_ration:
    :param number_of_elements:
    :param verbose:
    :return: Returns the discretized domain as a 1d numpy array with N number of points.
    '''

    #determine initial deltax
    delta_x = initial_deltax(len_domain=len_domain, expansion_ratio=expansion_ratio, number_of_elements=number_of_elements)

    domain = np.empty(number_of_elements+1)

    for i in range(0, len(domain)):
        if i == 0:
            domain[i] = 0
        elif i == 1:
            domain[i] = 0 + delta_x
        else:
            domain[i] = domain[i-1] + ((expansion_ratio**(i-1)) * delta_x)




    if verbose:
        print(f"This is deltax {delta_x}")
        for d in domain:
            print(d)
        plt.scatter(domain, np.ones(len(domain)))
        plt.show()


    return domain, delta_x