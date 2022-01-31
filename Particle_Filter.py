import math
from scipy.stats import norm
import numpy as np
from statistics import variance as sample_var


class Heston_Particle_Filter(object):
    """ This object contains all states of the particles. This object can also be updated to calculate
    the log likelihood given the initial parameters given to the class

    Attributes:
    -----------
    n : int : number of particles

    Observations = The observed log of returns

    h = Our Delta t Variable calculated as 1/252 (since 252 trading days in a year)

    Param = 1 x 5 Vector containing the values of:
            [Mu, Kappa, Theta, Xi, Rho] in this specific order
            where Mu = Mean drift of the asset
            Kappa = Reversion Speed
            Theta = Reversion Level
            Xi = Volatility of the Volatility
            Rho = Correlation between the Brownian Motions

    Class Methods:
    -------------
    State_mean : arr[float] : Conditional Mean of the current internal state

    State_vol : arr[float] : Conditional Volatility of the current internal state

    Obs_mean : arr[float] : Conditional Mean of the current observation

    Obs_vol : arr[float] : Conditional Volatility of the current observation

    resample : arr[float] : resamples from particles given their weights

    update : arr[float] : updates the particles and the weights according to the Heston Model
    """

    def __init__(self, n, observations, h=1 / 252, Param=[0.1, 1, 2, 3, 0.1]):
        # Parameters of the particle filter
        self.mu = np.array(Param[0])
        self.kappa = np.array(Param[1])
        self.theta = np.array(Param[2])
        self.xi = np.array(Param[3])
        self.rho = np.array(Param[4])
        self.delta_t = np.array(h)
        self.log_like = 0

        self.obs_len = len(observations)
        self.observations = observations
        self.particle_count = n
        self.state = 1
        self.eta = np.random.rand(self.particle_count)  # the brownian motion / noise added to The returns equation
        self.epsilon = np.random.rand(
            self.particle_count)  # the brownian motion / noise added to the Volatility equation
        self.INIT_vol = sample_var(observations)

        # initialize  matrices to 0. All matrices have shape [observation count, particle count]
        self.particles_matrix = np.zeros((self.obs_len, self.particle_count))
        # initialize the first column of particles to a uniform distribution between [initial_volatility/2 , initial_volatility *2]
        self.particles_matrix[0, :] = np.random.uniform(self.INIT_vol / 2, self.INIT_vol * 2, (self.obs_len, 1))[0, :]
        self.shape_checker(self.particles_matrix, (self.obs_len, self.particle_count), "particles matrix")

        self.weights_matrix = np.zeros((self.obs_len, self.particle_count))
        # initialize the first column of weights to (1/count of particles)
        self.weights_matrix[0, :] = 1 / self.particle_count
        self.shape_checker(self.weights_matrix, (self.obs_len, self.particle_count), "weights matrix")

    def shape_checker(self, np_array, expected_shape: tuple, array_name):
        if np_array.shape != expected_shape:
            raise ValueError(f"{array_name} has mutated, expected{expected_shape}, received {np_array.shape}")

    def RELU(self, x):  # volatility is strictly non negative, so we take relu of it
        return np.maximum(0, x)

    def internal_state_mean(self):
        temp = (self.particles_matrix[self.state - 1, :] + self.kappa * (
                self.theta - self.particles_matrix[self.state - 1, :])).reshape(1, self.particle_count)
        self.shape_checker(temp, (1, self.particle_count), "internal_state_mean")
        return temp

    def internal_state_vol(self):
        temp = (self.xi * np.sqrt(self.delta_t) * self.particles_matrix[self.state - 1, :] * self.eta[
            self.state]).reshape(1, self.particle_count)
        temp = self.RELU(temp)
        self.shape_checker(temp, (1, self.particle_count), "internal_state_vol")
        return temp

    def observation_mean(self):
        temp = (self.observations[self.state - 1] + (self.mu - self.particles_matrix[self.state, :] * 0.5)).reshape(1,
                                                                                                                    self.particle_count)
        self.shape_checker(temp, (1, self.particle_count), "Observation Mean")
        return temp

    def observation_vol(self):
        # creating correlated brownian motions between two equations using epsilon, eta and rho
        temp = (np.sqrt(self.particles_matrix[self.state, :]) *
                (self.rho * self.epsilon[self.state] + np.sqrt(1 - self.rho ** 2) * self.eta[self.state])
                ).reshape(1, self.particle_count)
        temp = self.RELU(temp)
        self.shape_checker(temp, (1, self.particle_count), "Observation Vol")
        return temp

    def resample(self):
        # if all weights are zero, replace them with 1/count of particles
        temp_weight_sum = sum(self.weights_matrix[self.state, :])
        if temp_weight_sum == 0:
            self.weights_matrix[self.state, :] = (1 / self.particle_count)

        # Normalizing the weights
        self.weights_matrix[self.state, :] /= temp_weight_sum

        # resampling form the particles according to their weights
        self.particles_matrix[self.state, :] = np.random.choice(self.particles_matrix[self.state, :],
                                                                self.particle_count,
                                                                p=self.weights_matrix[self.state, :])
        # reset the previous weights to (1/number of particles)
        self.weights_matrix[self.state, :] = (1 / self.particle_count)

    def update(self):
        while self.state < self.obs_len:
            # we update our volatility using the volatility and returns from previous states
            self.particles_matrix[self.state:, ] = self.internal_state_mean() + self.internal_state_vol()
            # price update:
            predicted_price = self.observation_mean() + self.observation_vol()

            # updating the weights matrix based on predicted prices and observed prices. / Assuming Gaussian
            temp = (- (self.observations[self.state] - predicted_price) ** 2) / 2
            self.weights_matrix[self.state, :] = (1 / math.sqrt(2 * math.pi)) * np.exp(temp)

            self.log_like += math.log(np.mean(self.particles_matrix[self.state, :]))

            # Re sampling form the particles based on their generated weights
            self.resample()

            self.state += 1

