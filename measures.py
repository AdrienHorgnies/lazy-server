from typing import Dict, List

import numpy as np
from numpy import ndarray


def compute_measures(arrivals: ndarray, services: ndarray, completions: ndarray, states: ndarray):
    """
    Compute statistical measures from the results of a single simulation

    :param arrivals: the times of arrivals
    :param services: the durations of the services
    :param completions: the times of completion of services
    :param states: the states of the server at arrivals
    :return: a dictionary containing the service_mean, sojourn_mean, p_off, p_setup and p_on
    """
    job_count = len(arrivals)

    if len(arrivals) == 0:
        return {
            'service_mean': 0,
            'sojourn_mean': 0,
            'p_off': 1,
            'p_setup': 0,
            'p_on': 0,
            'utilization': 0,
        }

    return {
        'service_mean': services.mean(),
        'sojourn_mean': (completions - arrivals).mean(),
        'p_off': (states == 'off').sum() / job_count,
        'p_setup': (states == 'setup').sum() / job_count,
        'p_on': (states == 'on').sum() / job_count,
        'utilization': services.sum() / completions[-1],
    }


def append_measures(base: Dict[str, List[float]], extension: Dict[str, float]):
    """
    Append the measures from extension to the list of measures from base

    :param base: a dictionary containing a list of measures for each type of measure
    :param extension: a dictionary containing a single value for each type of measure
    :return: None
    """
    for measure, value in extension.items():
        base[measure].append(value)


def mean_measures(measures_list: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Aggregate the list of values of each measure into its mean

    :param measures_list: a dictionary containing a list of measures for each type of measure
    :return: a dictionary containing the mean of all measures for each type of measure
    """
    return {
        measure: np.mean(values)
        for measure, values
        in measures_list.items()
    }


def mean_sojourn_time(_lambda, mu, rho, theta):
    """
    Compute the theoretical mean sojourn time

    :param _lambda: arrival rate
    :param mu: service rate
    :param rho: arrival rate / service rate
    :param theta: setup rate
    :return: theoretical mean sojourn time
    """
    return 1 / mu / (1 - rho) + 1 / theta


def compute_p_setup(_lambda, rho, theta):
    """
    Compute the theoretical value of P_SETUP

    :param _lambda: arrival rate
    :param rho: arrival rate / service rate
    :param theta: setup rate
    :return: theoretical value of P_SETUP
    """
    return (1 - rho) / (theta / _lambda + 1)


def compute_p_off(_lambda, rho, theta):
    """
    Compute the theoretical value of P_OFF

    :param _lambda: arrival rate
    :param rho: arrival rate / service rate
    :param theta: setup rate
    :return: theoretical value of P_SETUP
    """
    return (1 - rho) / (1 + _lambda / theta)
