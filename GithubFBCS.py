from math import inf
import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint 

'''The DE algorithm is a population-based heuristic algorithm [1] used to optimize non-linear and non-differentiable 
    continuous space functions. The DE algorithm uses three processes known as mutation, crossover, and selection to 
    optimize the functions. 

    [1] R. Storn, K. Price, “Differential Evolution – A Simple and Efficient
        Heuristic for global Optimization over Continuous Spaces,” J Glob.
        Optim., vol. 11, no. 4. Springer Science and Business Media LLC, pp.
        341–359, 1997.

    The DE algorithm has been employed for the optimization of some of the electronics and instrumentation systems, 
    and a list of the research papers, which have discussed the applications of the DE algorithm towards the 
    instrumentation area, is given below.

    [2] S. Sundararajan, K. N. Madhusoodanan, A. Abudhahir and G. Noble, "Implementation and Analysis of an 
    Evolutionary Optimized Non-Linear Function for Linearization of Thermo-Resistive Sensors," 6th International Conference 
    on Devices, Circuits and Systems (ICDCS), Coimbatore, India, 2022, pp. 74-79, doi: 10.1109/ICDCS54290.2022.9780706.

    [3] R. Yang, X. Li, R. Cong and J. Du, "A Novel Cylindricity Measurement Method for Large Workpiece Based on Improved Model and Algorithm,
        " in IEEE Transactions on Instrumentation and Measurement, vol. 73, pp. 1-11, 2024, Art no. 3000211, doi: 10.1109/TIM.2023.3331408.

    [4] Z. Qiu and Y. Zhang, "Three-Dimensional Low-Frequency Earthquake Monitoring Vibration Sensor Based on FBG,
        " in IEEE Transactions on Instrumentation and Measurement, vol. 73, pp. 1-9, 2024, Art no. 9510709, doi: 10.1109/TIM.2024.3406839.

    [5] S. Murugan, Dr. SP. Umayal, Dr. K. Srinivasan and M.Aruna. "Nonlinearity Error Compensation of Venturi 
        Flow Meter Using Evolutionary Optimization Algorithms" International Journal for Innovative Research 
        in Science & Technology, vol. 3, pp. 30-39, 2016.

    [6] T. I. Ha, J. -H. Lee and B. -K. Min, "Relative Positioning Error Minimization of the Dual-Robot System With Kinematic and Base Frame 
        Transformation Parameter Identification," in IEEE Access, vol. 11, pp. 54133-54142, 2023, doi: 10.1109/ACCESS.2023.3281193.
'''

'''Adding the temperature range with desired step size and their respective resistance values of the thermistor
    - T refer to the temperature in the range of 0 to 121 (exclusive) degree centigarde with a step size of 5 degree centigrade
    - RT refer to the thermistor resistance values for the above temperatures

    - As a case study, in this code, we used the thermistor (RT) charcteristics [2] with nominal resistance of 100000 ohms

    [7] DataSheet:NTCM-100K-B3950,“Specifications for NTC Thermistor”,
        Accessed: 20-11-2023, SR Passives, [Online] Available:
        https://www.tme.com/in/en/details/ntcm-100k-b3950/tht-measurement-ntc-thermistors/sr-passives/
    
    - As per the user requirements and specifications of the thermistor, T and RT can be changed.    
'''

T = np.array(list(range(0, 121, 5)))
RT = np.array([321140, 250886, 198530, 156407, 124692, 100000, 80650, 65395, 53300, 43659, 35840, 29713, 
                24681, 20592, 17253, 14517, 12265, 10404, 8859, 7573, 6498, 5595, 4836, 4194, 3649]) ## RT is in ohms

D = 2  # D refers to the dissipation constant of the thermistor [2]

def Calculate_VO1(T, RT, VR, GA, a, Y, Rp):
    '''
    defines the function that computes the mode-1 output voltage of the linearization circuit, VO1
    '''
    VO1 = (((GA*(VR))*(RT-(Rp)))/((2*(RT+(Rp))+(GA*(-Y)*(RT-(Rp))))))

    return(VO1)

def Calculate_VO2(T, RT, VR, GA, a, Y, Rp):
    '''
    defines the function that computes the mode-2 output voltage of the linearization circuit, VO2
    '''
    VO2 = ((((GA*(VR))*(RT-(Rp)))/((2*(RT+(Rp)))+(GA*(Y)*(RT-(Rp))))))
    return(VO2)


def Calculate_Vtheta(T, RT, VR, GA, a, Y, Rp):
    '''
    defines the function that computes the output voltage of the linearization circuit, Vtheta
    '''
    VO1 = (((GA*(VR))*(RT-(Rp)))/((2*(RT+(Rp))+(GA*(-Y)*(RT-(Rp))))))
    VO2 = ((((GA*(VR))*(RT-(Rp)))/((2*(RT+(Rp)))+(GA*(Y)*(RT-(Rp))))))

    Vtheta = (a*VO1)+((1-a)*VO2)

    return(Vtheta)

def BFL(T, RT, VR, GA, a, Y, Rp):
    '''
    Computation of the best fit line (BFL) for the output equation of the linearization circuit of the thermistor (say, V)
    '''
    VO1 = (((GA*(VR))*(RT-(Rp)))/((2*(RT+(Rp))+(GA*(-Y)*(RT-(Rp))))))
    VO2 = ((((GA*(VR))*(RT-(Rp)))/((2*(RT+(Rp)))+(GA*(Y)*(RT-(Rp))))))

    Vtheta = (a*VO1)+((1-a)*VO2)
    
    X = np.ones(shape = (len(T), 2))
    X[...,1] = np.array(T).transpose()
    b1 = np.linalg.inv(np.matmul(X.transpose(), X))
    b2 = np.matmul(b1, X.transpose())
    b  = np.matmul(b2, np.array(Vtheta).transpose())
    return (b.tolist())


def SC(T, RT, VR, GA, a, Y, Rp): 
    '''
    defines the function that computes the absolute value of the sensitivity of the thermistor
    '''
    return (BFL(T, RT, VR, GA, a, Y, Rp)[1])

def Calculate_SHE(T, RT, VR, GA, a, Y, Rp):
    '''
    defines the function that computes the self-heating error of the thermistor using dissipation constant
    '''

    VO1 = (((GA*(VR))*(RT-(Rp)))/((2*(RT+(Rp))+(GA*(-Y)*(RT-(Rp))))))
    VA21 = (VR)+(Y*VO1)

    '''IO1 is the current passing through the thermistor in mode-1'''
    IO1 = VA21/((RT+Rp))

    VO2 = ((((GA*(VR))*(RT-(Rp)))/((2*(RT+(Rp)))+(GA*(Y)*(RT-(Rp))))))
    VA22 = (VR)-(Y*VO2)

    '''IO2 is the current passing through the thermistor in mode-2'''
    IO2 = VA22/((RT+Rp))

    '''I is current passing through the thermistor'''
    I = (a*IO1)+((1-a)*IO2)

    '''P is the power dissipation across the thermistor'''
    P = (I*I)*RT 

    '''Pmax is the maximum power dissipation over 25 different temperatures'''
    Pmax = max(P) 

    '''SHE is defined as self-heating error'''
    SHE = Pmax/D 

    return(SHE)
def to_minimize(R):
    '''
    defines the optimization problem i.e, objective function that needs to be minimized (%NL) and computes the decision variable vector.
    R is a decision variable vector, R = [GA, N, a, Rp, Y], at which minimum %NL can be achieved
    '''

    GA = R[0]
    VR = R[1]
    a = R[2]
    Rp = R[3]
    Y = R[4]

    VO1 = (((GA*(VR))*(RT-(Rp)))/((2*(RT+(Rp))+(GA*(-Y)*(RT-(Rp))))))
    VO2 = ((((GA*(VR))*(RT-(Rp)))/((2*(RT+(Rp)))+(GA*(Y)*(RT-(Rp))))))

    Vtheta = (a*VO1)+((1-a)*VO2)

    b = BFL(T, RT, VR, GA, a, Y, Rp)

    '''c and m refers to the intercept and slope of the best-fit-line (BFL), respectively'''
    c, m = b

    '''y is the best-fit-line (BFL)'''
    y = m*T+c 
    
    '''defines the percentage non-linearity (%NL)'''
    NLn = max(abs(Vtheta-y))
    NLd = abs(max(Vtheta)-min(Vtheta))
    NL = 100 * (NLn/NLd)

    return(NL)


def constr_f1(x):
    '''
    defines the constraint function on the sensitivity i.e., absolute value of the slope of the BFL (abs(m))
    '''
    GA, VR, a, Rp, Y = x
    return np.array(abs(BFL(T, RT, VR, GA, a, Y, Rp)[1])) 

def constr_f2(x):
    '''
    defines the constraint function on the mode-1 output voltage, VO1
    '''

    GA, VR, a, Rp, Y = x
    return np.array(Calculate_VO1(T, RT, VR, GA, a, Y, Rp)) 

def constr_f3(x):
    '''
    defines the constraint function on the mode-1 output voltage, VO2
    '''

    GA, VR, a, Rp, Y = x
    return np.array(Calculate_VO2(T, RT, VR, GA, a, Y, Rp)) 

def constr_f4(x):
    '''
    defines the constraint function on the output voltage, Vtheta
    '''
    GA, VR, a, Rp, Y = x
    return Calculate_Vtheta(T, RT, VR, GA, a, Y, Rp)

def constr_f5(x):
    '''
    defines the constraint function on the self-heating erro (SHE)
    '''
    GA, VR, a, Rp, Y = x
    return Calculate_SHE(T, RT, VR, GA, a, Y, Rp)
  
'''lower bound on the sensitivity as 0.01 V/degree centigrade i.e., 10 mV/degree centigrade'''  
nlc1 = NonlinearConstraint(constr_f1, 0.01, inf)

'''lower and upper bounds on the mode-1 output voltage, VO1 are [-4,4]
    - Based on the power supply rails and input voltage range of the ADC, the above limits can be changed.
'''
nlc2 = NonlinearConstraint(constr_f2, -4, 4) 

'''lower and upper bounds on the mode-2 output voltage, VO2 are [-4,4]
    - Based on the power supply rails and input voltage range of the ADC, the above limits can be changed.
'''
nlc3 = NonlinearConstraint(constr_f3, -4, 4) 

'''lower and upper bounds on the output voltage, Vtheta are [-4,4]
    - Based on the power supply rails and input voltage range of the ADC, the above limits can be changed.
'''
nlc4 = NonlinearConstraint(constr_f4, -4, 4) 

'''upper bound on the self-heating error (SHE) as 0.05 degree centigrade'''
nlc5 = NonlinearConstraint(constr_f5, 0, 0.05)

'''nlc ensures that all five aforementioned non-linear constraints are satisfied'''
nlc = (nlc1, nlc2, nlc3, nlc4, nlc5)

'''
Lower and upper limit/bound of search space for each optimal parameters
    - search space for the gain of the instrumentation amplifier (A1) GA is [1,1000]
    - search space for the reference voltage, VR is (0,2.5]
    - search space for the duty-ratio, a is [0.01,0.99]
    - search space for the bridge resistor, Rp is [1,1000000] ohms
    - search space for the gain of the instrumentation amplifier (A2), Y is (0,1000]
'''

bounds = [(1,1e3), (0,2.5), (0.01,0.99), (1,1e6), (0,1e3)]

'''
Differential evolution algorithm minimizes the function named "to_minimize", with the defined bounds and constraint
    Parameters:
    - maxiter refer to the maximum iterations for the framed constrained optimization problem
    - polish can be True or False. for the optimization problems with many constraints, polishing can take a long time
    due to the Jacobian computations.
    - integrality - For each decision variable, a boolean value indicating whether the decision variable is constrained to integer values.
    if the integrality is 1 for the decision variable, only integer values lying between the lower and upper bounds are used for solving
    the optimization problem.
'''
result = differential_evolution(to_minimize, bounds, constraints=(nlc), maxiter = 1000, polish = False, integrality=[0,0,0,0,0])

'''prints the result'''
GA, VR, a, Rp, Y = result.x

'''prints the below function returnable values'''
print(Calculate_VO1(T, RT, VR, GA, a, Y, Rp))
print(Calculate_VO2(T, RT, VR, GA, a, Y, Rp))
print(Calculate_Vtheta(T, RT, VR, GA, a, Y, Rp))
print(Calculate_SHE(T, RT, VR, GA, a, Y, Rp))
print(result)
