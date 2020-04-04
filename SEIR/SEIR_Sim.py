from dataclasses import dataclass
import numpy as np
#import sys
import scipy.integrate
import matplotlib.pyplot as plt
import scipy.ndimage.interpolation

#sys.path.append(".")

from parameters import DiseaseParams, SimOpts, PlotOpts, Country_Info




@dataclass
class Store_Results:
    """
    Data class to store simulation results
    Note: Dataclass decorator takes care of self initialization such as __init__, and __repr__, see python documentation
    E.g.
    @dataclass
    class Person:
        name: str
        age: int

    p = Person('John Doe', 34)
    print(p)
    output:
    Person(name='John Doe', age=34)
    """
    T: np.array # timesteps (days), used for ODE input
    S: np.array # Susceptible at given timestep
    E: np.array # Exposed at given timestep
    I: np.array # Infected at given timestep
    R: np.array # Recovered at given timestep
    D: np.array = np.zeros(0) # Deaths at given timestep
    F: np.array = np.zeros(0) # Found at given timestep
    H: np.array = np.zeros(0) # Hospitalized at given timestep
    P: np.array = np.zeros(0) # Probability of random person being infected at given timestep

class SEIR_Model:
    """
    SEIR Model
    """

    def __init__(self, country_name: str, n_pop: int):
        """
        :param country_name: name e.g. "Switzerland"
        :param n_pop: number of habitants
        """
        self.country_name = country_name
        self.n_pop = n_pop

    def model_seir(self, t: float, state: np.ndarray, disease_parameters: DiseaseParams, simulation_parameters: SimOpts):
        """
        :param t: time-step (days)
        :param state: vector of ODE state variables [S,E,I,R]
        :param disease_parameters: disease parameters
        :param simulation_parameters: simulation parameters
        :return: dS, dE, dI, dR all of type (float)
        """
        
        N = self.n_pop #population
        S, E, I, R = state

        if simulation_parameters.lockdown is True:
            if t <= simulation_parameters.lockdown_delay:
                beta = disease_parameters.beta_init
            else:
                beta = disease_parameters.beta_lock
        else:
            beta = disease_parameters.beta_init

        sigma = disease_parameters.sigma
        gamma = disease_parameters.gamma

        dS = - beta * S * I / N
        dE = beta * S * I / N - sigma * E
        dI = sigma * E - gamma * I
        dR = gamma * I

        return dS, dE, dI, dR

    def run_seir(self, disease_parameters: DiseaseParams, simulation_parameters: SimOpts):
        """
        Solve epidemical ODE's
        :param disease_parameters: disease parameters
        :param simulation_parameters: simulation parameters
        return: array with time development
        """

        T = np.arange(simulation_parameters.sim_length) #array with equally spaced time steps (days)
        Y0 = [self.n_pop - simulation_parameters.initial_exposed, simulation_parameters.initial_exposed, 0, 0] #S, E, I, R

        print('Test1', disease_parameters)
        print('Test2', simulation_parameters)
        Y_results = scipy.integrate.solve_ivp(self.model_seir, t_span=[T[0], T[-1]], 
                                              y0=Y0, model='RK45', args=(disease_parameters, simulation_parameters), t_eval=T)

        S, E, I, R = Y_results.y #unpack results

        raw_results = Store_Results(T=T, S=S, E=E, I=I, R=R)

        #Without additional parameters calculated
        #parsed_results = raw_results

        #With additional model parameters
        parsed_results = self._parse_results(raw_results, disease_parameters, simulation_parameters)

        return parsed_results


    def _parse_results(self, results: Store_Results, disease_parameters: DiseaseParams, simulation_parameters: SimOpts):
        """
        Private helper function to calculate death rate given some additional parameter assumption
        """
        T, S, E, I, R = results.T, results.S, results.E, results.I, results.R

        F = I*disease_parameters.find_ratio #calculate infected found at given time
        H = I*disease_parameters.rate_icu*disease_parameters.time_hospital/disease_parameters.time_infected #is that correct?
        P = I/self.n_pop*100 #Probability of a random person to be infected

        #Estimate death population from recovered population with the influence of the hospital beds
        D = np.arange(simulation_parameters.sim_length) #initialize death array
        
        R_prev = 0
        D_prev = 0

        for i, t in enumerate(results.T):
            I_FR = disease_parameters.rate_fatality_0 if H[i] <= simulation_parameters.icu_beds else disease_parameters.rate_fatality_1
            D[i] = D_prev + I_FR*(R[i] - R_prev)
            R_prev = R[i]
            D_prev = D[i]

        if simulation_parameters.add_delays is True:
            F = self.delay(F, disease_parameters.lag_symptom_to_hosp + disease_parameters.lag_testing + disease_parameters.lag_communication)
            I = self.delay(H, disease_parameters.lag_symptom_to_hosp)
            D = self.delay(D, disease_parameters.time_hospital + disease_parameters.lag_communication)

        results.P = P
        results.F = F
        results.H = H
        results.D = D

        return results

    @staticmethod
    def delay(array: np.ndarray, days_shift: int):
        """
        Use scipy method to shift values with spline interpolation
        """
        return scipy.ndimage.interpolation.shift(array, days_shift, cval=0)


if __name__ == '__main__':

    disease_parameters = DiseaseParams()
    simulation_paramters = SimOpts()
    country_parameters = Country_Info()
  
    model = SEIR_Model(country_parameters.country_name, country_parameters.country_population)
    results = model.run_seir(disease_parameters, simulation_paramters)

    plt.plot(results.T,results.S, label='Susceptible')
    plt.plot(results.T,results.E, label='Exposed')
    plt.plot(results.T,results.I, label='Infected')
    plt.plot(results.T,results.R, label='Recovered')
    plt.plot(results.T,results.D, label='Deaths')
    plt.legend()
    plt.show()

    print(results.D[-1])

