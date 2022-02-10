# Particle Filter Heston Model
 This Heston particle filter class calculates the log liklihood values of the initial values given to the class for the heston model (heston, 1993) using particle filters.

Heston model is an extremly important Stochastic volatility model. Many of the current advanced models are variations of this model. For example, we arrive at the bates model simply allows stochastic jumps for the heston model.

A good introduction to the heston model can be found at: https://en.wikipedia.org/wiki/Heston_model

While particle filters can be very complicated to explain, a very simple introduction to them would be as follows:

In Particle filter we generate a cloud of particles around an initial guess.
Then we generate a likelihood that these particles gave rise to our observed returns. These
likelihoods form the weights of our particles. We then resample from these particles according to
their weights. Naturally, we are more likely to pick better particles since they have higher weights.
This will result in us generating a better cloud of particles in our next observation.

The main issues known about particle filters are:
1) particle filters often give rise to a non smooth likelihood surface, therefore minimizing the likelihood function could be challenging
2) specifically with Heston model (and other stochastic volatility models), particle filters could give rise to negative values for the volatility terms for some particles, which is clearly not possible. in this implementation of the Particle Filters, we simply took the Max(0, Volatility) to ensure we have non negative particles.
