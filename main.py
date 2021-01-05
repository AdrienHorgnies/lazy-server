import argparse
import matplotlib.pyplot as plt
from numpy import ndarray
import numpy as np
from numpy.random import SeedSequence, SFC64, Generator
from typing import List, Tuple, Callable


def simulation(get_inter_arrival: Callable, get_service_duration: Callable, get_start_duration: Callable, tau) \
        -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Run a single simulation of the system, from t = 0 to t = tau

    :param get_inter_arrival: function without argument that produces a variate for the interarrival time
    :param get_service_duration: function without argument that produces a variate for the service duration
    :param get_start_duration: function without argument that produces a variate for the server start duration
    :param tau: the limit for the last event arrival
    :return: a tuple of the arrival times, the service durations, the completion times and the state of the server at
    client arrival
    """
    arrivals = []
    services = []
    completions = []
    states = []

    def arrival():
        return t + get_inter_arrival()

    def get_start():
        return t + get_start_duration()

    def completion():
        service = get_service_duration()
        services.append(service)
        events['completion'] = t + service
        completions.append(events['completion'])

    t = 0
    t = arrival()
    events = {
        'arrival': t,
        'completion': float('inf'),
        'start': float('inf'),
    }
    in_node = 0

    while t < tau or in_node:
        next_event = min(events, key=events.get)
        t = events[next_event]

        if next_event == 'arrival':
            arrivals.append(t)
            in_node += 1

            next_arrival = arrival()
            events['arrival'] = next_arrival if next_arrival <= tau else float('inf')

            if events['completion'] != float('inf'):
                states.append('on')
            elif events['start'] != float('inf'):
                states.append('setup')
            else:
                states.append('off')
                events['start'] = get_start()
        elif next_event == 'start':
            events['start'] = float('inf')
            completion()
        elif next_event == 'completion':
            in_node -= 1
            if in_node > 0:
                completion()
            else:
                events['completion'] = float('inf')

    return np.array(arrivals[:len(completions)]), np.array(services), np.array(completions), np.array(states)


def exponential_simulation(generators: List[Generator], _lambda, mu, theta, tau):
    """

    :param generators:
    :param _lambda: arrival rate
    :param mu: service rate
    :param theta: server starting rate
    :param tau: the limit for the last event arrival

    :return: a tuple of the arrival times, the service durations and the completion times
    """

    def get_inter_arrival():
        return generators[0].exponential(1 / _lambda)

    def get_service_duration():
        return generators[1].exponential(1 / mu)

    def get_start_duration():
        return generators[2].exponential(1 / theta)

    return simulation(get_inter_arrival, get_service_duration, get_start_duration, tau)


def erlang_simulation(generators: List[Generator], _lambda, mu, theta):
    """

    :param generators:
    :param _lambda: arrival rate
    :param mu: service rate
    :param theta: server starting rate
    :return:
    """
    pass


def compute_measures(arrivals: ndarray, services: ndarray, completions: ndarray, states: ndarray):
    job_count = len(arrivals)

    return {
        'service_mean': services.mean(),
        'sojourn_mean': (completions.sum() - arrivals.sum()) / job_count,
        'p_off': (states == 'off').sum() / job_count,
        'p_setup': (states == 'setup').sum() / job_count,
        'p_on': (states == 'on').sum() / job_count,
    }


def main():
    description = 'Produce the different results presented in the project report. ' \
                  'The project report uses the seed XXXXX and it took XXXX minutes to complete.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('seed', nargs='?', type=int, help='seed to initialize the random generator')
    args = parser.parse_args()

    if args.seed:
        seed_seq = SeedSequence(args.seed)
    else:
        seed_seq = SeedSequence()
    print('Seed : ', seed_seq.entropy)
    # Making more generators than required to anticipate possible further needs
    generators = [Generator(SFC64(stream)) for stream in seed_seq.spawn(20)]

    _lambda = 1
    mu = 1 / 0.5
    theta = 0.2
    tau = 10000

    arrivals, services, completions, states = exponential_simulation(generators[:3], _lambda, mu, theta, tau)
    measures = compute_measures(arrivals, services, completions, states)

    rho_eq = _lambda / mu

    rho_def = _lambda * measures['service_mean']
    sojourn_mean_theo = 1 / (mu - _lambda) + 1 / theta
    p_off_theo = (1 - rho_def) / (1 + _lambda / theta)
    p_setup_theo = (1 - rho_def) / (theta / _lambda + 1)
    p_on_theo = 1 - p_off_theo - p_setup_theo

    assert np.isclose(rho_def, rho_eq, rtol=0.05), (rho_def, rho_eq)
    assert np.isclose(sojourn_mean_theo, measures['sojourn_mean'], rtol=0.05), (sojourn_mean_theo, measures['sojourn_mean'])
    assert np.isclose(p_off_theo, measures['p_off'], rtol=0.05), (p_off_theo, measures['p_off'])
    assert np.isclose(p_setup_theo, measures['p_setup'], rtol=0.05), (p_setup_theo, measures['p_setup'])
    assert np.isclose(p_on_theo, measures['p_on'], rtol=0.05), (p_on_theo, measures['p_on'])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
