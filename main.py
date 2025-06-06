import numpy as np
import scipy.integrate as integ
import scipy.interpolate as interp
import matplotlib.pyplot as plt


#############
# CONSTANTS #
#############

# Define transition rates
LAMBDA = 1
MU = 5

# Define generator matrix

Q = np.array([
    [-3*LAMBDA, MU, 0, 0],
    [3*LAMBDA, -(2*LAMBDA + MU), MU, 0],
    [0, 2*LAMBDA, -(LAMBDA+MU), MU],
    [0, 0, LAMBDA, -MU]
])

# Define Markov chain states and transitions

W3, W2, W1, W0 = 0, 1, 2, 3
TRANSITIONS = {
    W3: [
        (W2, 3*LAMBDA)
    ],
    W2: [
        (W1, 2*LAMBDA),
        (W3, MU)
    ],
    W1: [
        (W0, LAMBDA),
        (W2, MU)
    ],
    W0: [
        (W1, MU)
    ]
}

# Define additional simulation parameters
SIM_TIME = 2
REPETITIONS = 5000

#############
# FUNCTIONS #
#############

# Define master equations for the TMR

def master_equations(p, t):
    return np.matmul(Q, p)

# Define Markov chain single step

def markov_chain_step(state):
    available_transitions = TRANSITIONS[state]
    
    possible_next_state = list(map(lambda x: x[0], available_transitions))
    rates = list(map(lambda x: x[1], available_transitions))

    total_rate = np.sum(rates)
    delta_time = np.random.exponential(1 / total_rate)

    weights = rates / total_rate
    next_state = np.random.choice(possible_next_state, p=weights)

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

    time.append(simulation_time)
    state.append(state[-1])

    interpolator = interp.interp1d(time, state, kind='previous')
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

    TIME = np.linspace(0, SIM_TIME, 1000)
    
    ########## Numerical solution of master equations

    result = integ.odeint(master_equations, [1, 0, 0, 0], TIME)

    for i in range(4):
        plt.plot(TIME, result[:,i], color='black')

    availability = result[:,0] + result[:,1]

    plt.plot(TIME, availability, color='black')

    ########## Markov chain simulation
    runs = []

    for _ in range(REPETITIONS):
        state = simulate_markov_chain(W3, TIME)
        state = list(map(state_mapping, state))
        runs.append(state)

    runs = np.array(runs)

    probabilities = np.mean(runs, axis=0)

    for i in range(4):
        plt.plot(TIME, probabilities[:,i], label= str(i))

    availability = probabilities[:,0] + probabilities[:,1]

    plt.plot(TIME, availability, label='A')

    ########## Plots

    plt.legend()
    plt.xlabel('Time (years)')
    plt.ylabel('Probabilities')
    plt.show()