import numpy as np
import scipy.integrate as integ
import scipy.interpolate as interp
import matplotlib.pyplot as plt


#############
# CONSTANTS #
#############

# Define transition rates

lam = 2
delta = 1
mu = 3

# Define generator matrix
Q = np.array([
    [-(lam + 2*delta), mu, 0, 0],
    [(lam + 2*delta), -(lam + delta + mu), mu, 0],
    [0, (lam + delta), -(lam + mu), 0],
    [0, 0, lam, 0],
])

# Define Markov chain states and transitions
W3, W2, W1, W0 = 0, 1, 2, 3

TRANSITIONS = {
    W3: [
        (W2, lam + 2*delta)
    ],
    W2: [
        (W3, mu),
        (W1, lam + delta)
    ],
    W1: [
        (W2, mu),
        (W0, lam)
    ],
    W0: [
        (W1, 1e-64)
    ],
}

# Define additional simulation parameters
SIM_TIME = 5

REPETITIONS = 5000

failure_time = []

#############
# FUNCTIONS #
#############

# Define master equations for the TMR
def master_equations(p, t):
    return np.matmul(Q, p)


# Define Markov chain single step
def markov_chain_step(current_state):
    available_transitions = TRANSITIONS[current_state]

    possible_next_states = list(map(lambda x: x[0], available_transitions))
    rates = list(map(lambda x: x[1], available_transitions))

    total_rate = np.sum(rates)
    delta_time = np.random.exponential(1 / total_rate)

    weights = rates / total_rate
    next_state = np.random.choice(possible_next_states, p = weights)

    return (delta_time, next_state)


# Complete Markov chain simulation
def simulate_markov_chain(initial_state, time_grid):
    current_time = time_grid[0]
    current_state = initial_state

    simulation_time = time_grid[-1]

    time = []
    state = []

    while current_time < simulation_time:
        time.append(current_time)
        state.append(current_state)

        delta_time, next_state = markov_chain_step(current_state)

        current_time += delta_time
        current_state = next_state
        if current_state == W0:
            failure_time.append(current_time)
            
    
    time.append(simulation_time)
    state.append(state[-1])    

    interpolator = interp.interp1d(time, state, kind = 'previous')
    state = interpolator(time_grid)

    return state


# State to vector conversion
def state_mapping(state):
    if state == W3:
        return [1, 0, 0, 0]
    elif state == W2:
        return [0, 1, 0, 0]
    elif state == W1:
        return [0, 0, 1, 0]
    else:
        return [0, 0, 0, 1]


################
# MAIN PROGRAM #
################

if __name__ == '__main__':
    time = np.linspace(0, SIM_TIME, 1000)

    ########## Numerical solution of master equations

    result = integ.odeint(master_equations, [1, 0, 0, 0], time)

    for i in range(4):
        plt.plot(time, result[:, i], color = 'black')
    
    availability = result[:, 0] + result[:, 1]

    plt.plot(time, availability, color = 'black')

    ########## Markov chain simulation

    runs = []
    for _ in range(REPETITIONS):
        state = simulate_markov_chain(W3, time)
        state = list(map(state_mapping, state))
        runs.append(state)

    runs = np.array(runs)

    probabilities = np.mean(runs, axis = 0)

    for i in range(4):
        plt.plot(time, probabilities[:, i], label = str(i))
    
    availability = probabilities[:, 0] + probabilities[:, 1]

    plt.plot(time, availability, label = 'A')

    ########## Plots

    plt.legend()

    plt.show()

    plt.figure()
    plt.hist(failure_time, bins=50, edgecolor = 'blue')
    plt.show()
