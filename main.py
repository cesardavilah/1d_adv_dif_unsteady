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

    N = 100
    DELTA_T = .001
    N_STEPS = 201
    checkpoint = 50  # INTERVAL TO SAVE HISTORY

    #discretize
    domain, delta_x = cds.discretize_domain(len_domain=L, expansion_ratio=EX_RATIO, number_of_elements=N, verbose=False)

    #initialize

    initial_guess = cds.initialize_domain(domain=domain, CONST_DICT=CONST_DICT, interpolation_method="linear", verbose=False)

    #analytical solution
    analytical_solution = cds.exact_solution(domain=domain, CONST_DICT=CONST_DICT)

    #calculate delta_t_marginal
    DELTA_T_MARGNIAL = rk.delta_t_marginal(delta_x=delta_x, CONST_DICT=CONST_DICT, verbose=True)

    DELTA_T_MARGNIAL_105 = DELTA_T_MARGNIAL * 1.05
    DELTA_T_MARGNIAL_95 = DELTA_T_MARGNIAL * .95

    TSEL = 95*DELTA_T_MARGNIAL
    TSEL_STEPS = 100
    TSEL_STEPS_95L = 500



    #use RK2

    rk_2_solution_DT105, history_DT105 = rk.runge_kutta(initial_guess=initial_guess, order=2, analytical_solution=analytical_solution, CONST_DICT=CONST_DICT, DELTA_T=DELTA_T_MARGNIAL_105, domain=domain, MAX_STEPS=N_STEPS, checkpoint=checkpoint, verbose=False)

    rk_2_solution_DT95, history_DT95 = rk.runge_kutta(initial_guess=initial_guess, order=2, analytical_solution=analytical_solution, CONST_DICT=CONST_DICT, DELTA_T=DELTA_T_MARGNIAL_95, domain=domain, MAX_STEPS=N_STEPS, checkpoint=checkpoint, verbose=False)



    #TSEL
    solutions_tsel = []
    histories_tsel = []#where 0=100, 1=200 ...denom=100+i*100

    for denom in [100, 200, 300, 400, 500]:
        # print(denom)
        rk_2_solution_TSEL, history_TSEL = rk.runge_kutta(initial_guess=initial_guess, order=2,
                                                                  analytical_solution=analytical_solution,
                                                                  CONST_DICT=CONST_DICT, DELTA_T=TSEL/denom,
                                                                  domain=domain, MAX_STEPS=TSEL_STEPS,
                                                                  checkpoint=checkpoint, verbose=False)
        solutions_tsel.append(rk_2_solution_TSEL)
        histories_tsel.append(history_TSEL)

    rk_s_solution_95L, history_95L = rk.runge_kutta(initial_guess=initial_guess, order=2, analytical_solution=analytical_solution, CONST_DICT=CONST_DICT, DELTA_T=TSEL/500, domain=domain, MAX_STEPS=TSEL_STEPS_95L, checkpoint=1, verbose=False)





    #*************************PLOTS********************************
    #CALCULATE ERROR
    error_95L = []
    for h in history_95L:
        er = np.abs(analytical_solution[95] - h[95])
        error_95L.append(er)

    plt.figure()
    plt.title("log-log error, rate of convergence")



    n2 = cds.power(np.linspace(0,TSEL_STEPS_95L, num=TSEL_STEPS_95L), -2)

    # plt.loglog(n2, , label="Relative Error Numerical solution Second order")
    plt.loglog(np.linspace(0,TSEL_STEPS_95L,num=TSEL_STEPS_95L), n2, label="Ideal slope")
    plt.loglog(np.linspace(0,TSEL_STEPS_95L, num=TSEL_STEPS_95L), error_95L, label='Error')
    plt.xlabel("Delta T")
    plt.ylabel("Error")
    plt.legend()
    plt.savefig(path_to_save_figs+"\log-log error, rate of convergence.png")
    # plt.show()






    plt.figure()
    plt.title("tsel solutions for delta_t= tsel/100, tsel/200 ...")
    for i in range(0, len(solutions_tsel)):
        plt.plot(domain, solutions_tsel[i], label=f"Solution for tsel/{100+(i*100)}")
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Phi")
    # plt.show()
    plt.savefig(path_to_save_figs+r"\\tsel solutions for delta_t= tsel100, tsel2200 ....png")



    #PLOT 105 MARGINAL
    plt.figure()
    plt.title("RK2 with 105% DT_MARGINAL")
    for i in range(0, len(history_DT105)):
        plt.plot(domain, history_DT105[i], label=f"Step {i*checkpoint}")


    plt.legend(loc="lower left",mode = "expand", ncol = 3)
    plt.xlabel("X")
    plt.ylabel("Phi")
    plt.savefig(path_to_save_figs+"\RK2 with 105% DT_MARGINAL.png")





    #PLOT 95 MARGINAL
    plt.figure()
    plt.title("RK2 with 95% DT_MARGINAL")
    for i in range(0, len(history_DT95)):
        plt.plot(domain, history_DT95[i], label=f"Step {i*checkpoint}")

    plt.plot(domain, analytical_solution, label="Analytical solution", color='Green', marker='^')
    plt.xlabel("X")
    plt.ylabel("Phi")
    # plt.xlim(4, 5)
    plt.legend()
    plt.savefig(path_to_save_figs+"\RK2 with 95% DT_MARGINAL.png")
    # plt.show()

