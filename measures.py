"""
Functions to manipulate and compute measures from the results of simulations and function to compute measures from
system parameter.
"""
from typing import Dict, List, Tuple

import numpy as np
import scipy.stats as st
from numpy import ndarray


def compute_measures(arrivals: ndarray, services: ndarray, completions: ndarray, states: ndarray):
    """
    Compute statistical measures from the results of a single simulation

    :param arrivals: the times of arrivals
    :param services: the durations of the services
    :param completions: the times of completion of services
    :param states: the states of the server at arrivals
    :return: a dictionary containing the mean_service, mean_sojourn, p_off, p_setup, p_on and utilization
    """
    job_count = len(arrivals)

    # Required for absurd parameters I used to isolate some problems in my simulation
    if len(arrivals) == 0:
        return {
            'mean_service': 0,
            'mean_sojourn': 0,
            'p_off': 1,
            'p_setup': 0,
            'p_on': 0,
            'utilization': 0,
        }

    return {
        'mean_service': services.mean(),
        'mean_sojourn': (completions - arrivals).mean(),
        'p_off': (states == 'off').sum() / job_count,
        'p_setup': (states == 'setup').sum() / job_count,
        'p_on': (states == 'on').sum() / job_count,
        'utilization': services.sum() / (completions[-1] - arrivals[0]),
    }


def append_measures(base: Dict[str, List[float]], extension: Dict[str, float]):
    """
    Append the measures from extension to the list of measures from base.
    base is modified in place

    :param base: a dictionary containing a list of values for each type of measure
    :param extension: a dictionary containing a single value for each type of measure
    :return: None
    """
    for measure, value in extension.items():
        base[measure].append(value)


def mean_measures(measures_list: Dict[str, List[float]], ref=None) -> Dict[str, float]:
    """
    Aggregate the list of values of each measure into its mean.
    If a reference is present for a measure, also add its 95% confidence interval, H_0 ttest and theoretical value
    with the prefixes lower_ci_ (CI lower bound), higher_ci_ (CI higher bound), test_ and theo_.

    H_0 ttest consists of H_0 : v = v_t, H_1 : v != v_t, with v the measure, v_t its theoretical value and alpha = 5%
    test is True is H_0 is accepted, else it's False.

    :param measures_list: a dictionary containing a list of measures for each type of measure
    :param ref: a dictionary with reference measure to test hypothesis sample against reference using t-student
    :return: a dictionary containing aggregated measures
    """
    measures = dict()
    for key, values in measures_list.items():
        measures[key] = np.mean(values)
        if ref and key in ref:
            measures['lower_ci_' + key], measures['higher_ci_' + key] = confidence(values, measures[key])
            test = st.ttest_1samp(values, ref[key]).pvalue > 0.05
            measures['test_' + key] = test
            measures['theo_' + key] = ref[key]

    return measures


def exp_sojourn_time(_lambda, mu, rho, theta):
    """
    Compute the theoretical mean sojourn time for the exponential system

    :param _lambda: arrival rate
    :param mu: service rate
    :param rho: arrival rate / service rate
    :param theta: setup rate
    :return: theoretical mean sojourn time
    """
    return 1 / mu / (1 - rho) + 1 / theta


def exp_p_setup(_lambda, rho, theta):
    """
    Compute the theoretical value of P_SETUP for the exponential system

    :param _lambda: arrival rate
    :param rho: arrival rate / service rate
    :param theta: setup rate
    :return: theoretical value of P_SETUP
    """
    return (1 - rho) / (theta / _lambda + 1)


def exp_p_off(_lambda, rho, theta):
    """
    Compute the theoretical value of P_OFF for the exponential system

    :param _lambda: arrival rate
    :param rho: arrival rate / service rate
    :param theta: setup rate
    :return: theoretical value of P_SETUP
    """
    return (1 - rho) / (1 + _lambda / theta)


def erlang_sojourn_time(_lambda, n, b, rho, theta):
    """
    Compute the theoretical mean sojourn time for the Erlang system

    :param _lambda: arrival rate
    :param n: form of Erlang
    :param b: intensity of Erlang
    :param rho: lambda * n * b
    :param theta: setup rate
    :return: theoretical value for the mean sojourn time
    """
    assert theta == 0.4, 'precomputed values r_b and r_t are only valid for theta = 0.4'
    e_b = n * b
    e_t = 1 / theta
    # E[B^2] = (np.random.gamma(n, b, 5 * 10**7) ** 2).mean()
    r_b = 1.1000267388299316 / (2 * e_b)
    # E[T^2] = (np.random.exponential(1 / theta, 5 * 10**7) ** 2).mean()
    r_t = 12.501505592571696 / (2 * e_t)
    # r_t = theta ** 2 / (2 * e_t)
    wait = rho * r_b / (1 - rho) + \
           e_t / (1 + _lambda * e_t) + \
           e_t * r_t / (1 / _lambda + e_t)
    service = e_b
    return wait + service


def erlang_p_setup(_lambda, n, b, rho, theta):
    """
    Compute theoretical P_SETUP for the Erlang system

    :param _lambda: arrival rate
    :param n: form of Erlang
    :param b: intensity of Erlang
    :param rho: lambda * n * b
    :param theta: setup rate
    :return: theoretical value for the mean sojourn time
    """
    return exp_p_setup(_lambda, rho, theta)


def erlang_p_off(_lambda, n, b, rho, theta):
    """
    Compute theoretical P_OFF for the Erlang system

    :param _lambda: arrival rate
    :param n: form of Erlang
    :param b: intensity of Erlang
    :param rho: lambda * n * b
    :param theta: setup rate
    :return: theoretical value for the mean sojourn time
    """
    return exp_p_off(_lambda, rho, theta)


def confidence(sample, mean=None) -> Tuple[float, float]:
    """
    Compute the 95% confidence interval for the provided sample using t-student method

    :param sample: the sample
    :param mean: the mean of the sample
    :return: the lower bound and higher bound of the confidence interval
    """
    return st.t.interval(0.95, len(sample) - 1, loc=mean or np.mean(sample), scale=st.sem(sample))
