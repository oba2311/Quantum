#quantum_monte_carlo.py
"""
This script includes Variational MC.
Adopted from: 
https://galeascience.wordpress.com/2016/04/27/markov-chain-monte-carlo-sampling/
"""

import numpy as np
 
def Metroplis_algorithm(N, m, dr):
    ''' A Markov chain is constructed, using the
    Metropolis algorithm, that is comprised of
    samples of our probability density: psi(x,y).
 
    N - number of random moves to try
    m - will return a sample when i%m == 0
        in the loop over N
    dr - maximum move size (if uniform),
         controls the acceptance ratio '''
    # we'll want to return the average
    # acceptance ratio
    a_total = 0
 
    # sample locations will be stored in a list
    samples = []
 
    # get the starting configuration
    # and sample probability distribution
    # we'll start at r=(0,0)
    r_prime = np.zeros(2)
    p_prime = psi(r_prime[0], r_prime[1])
 
    for i in range(N):
        # propose a random move: r'-> r
        r = r_prime + np.random.uniform(-dr,dr,
                                        size=2)
        p = psi(r[0], r[1])
 
        # calculate the acceptance ratio
        # for the proposed move
        a = min(1, p/p_prime)
        a_total += a
 
        # check for acceptance
        p_prime, r_prime = check_move(p_prime, p,
                                      r_prime, r)
 
        if i%m == 0:
            samples.append(r_prime)
 
    return np.array(samples), a_total/N*100.0
 
def check_move(p_prime, p, r_prime, r):
    ''' The move will be accepted or rejected
        based on the ratio of p/p_prime and a
        random number. '''
 
    if p/p_prime >= 1:
        # accept the move
        return p, r
 
    else:
        rand = np.random.uniform(0, 1)
        if p/p_prime + rand >= 1:
            # accept the move
            return p, r
        else:
            # reject the move
            return p_prime, r_prime


import matplotlib.mlab as mlab
 
def psi(x, y):
    ''' Our probability density function is the addition
        of two 2D Gaussians with different shape. '''
    g1 = mlab.bivariate_normal(x, y, 2.0, 2.0, -5, -5, 0)
    g2 = mlab.bivariate_normal(x, y, 0.5, 5.0, 10, 10, 0)
    return g1 + g2

N, m, dr = 50000, 10, 3.5
samples, a = Metroplis_algorithm(N, m, dr)
psi(samples, a)