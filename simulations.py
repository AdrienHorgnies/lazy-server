"""
Functions to simulate the system
"""
from typing import Callable, Tuple, List

import numpy as np
from numpy import ndarray
from numpy.random import Generator


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

    while events['arrival'] < tau or in_node:
        next_event = min(events, key=events.get)
        t = events[next_event]

        if next_event == 'arrival':
            arrivals.append(t)
            in_node += 1

            next_arrival = arrival()  # check t before this happens
            events['arrival'] = next_arrival if next_arrival < tau else float('inf')

            if events['completion'] < float('inf'):
                states.append('on')
            elif events['start'] < float('inf'):
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

    arrivals = np.array(arrivals)
    second_half_filter = arrivals > tau / 2

    arrivals = arrivals[second_half_filter]
    services = np.array(services)[second_half_filter]
    completions = np.array(completions)[second_half_filter]
    states = np.array(states)[second_half_filter]

    return arrivals, services, completions, states


def exponential_simulation(generators, _lambda, mu, theta, tau):
    """
    Run a single simulation of the system, uses exponential variables for arrival, service and server setup

    :param generators: a list of three generators
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