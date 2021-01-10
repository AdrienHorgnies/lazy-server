"""
Functions to manipulate and compute measures from the results of simulations and function to compute measures from
system parameter.
"""
from typing import Dict, List

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
    :return: a dictionary containing the mean_service, mean_sojourn, p_off, p_setup and p_on
    """
    job_count = len(arrivals)

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
    Append the measures from extension to the list of measures from base

    :param base: a dictionary containing a list of measures for each type of measure
    :param extension: a dictionary containing a single value for each type of measure
    :return: None
    """
    for measure, value in extension.items():
        base[measure].append(value)


def mean_measures(measures_list: Dict[str, List[float]], ref=None) -> Dict[str, float]:
    """
    Aggregate the list of values of each measure into its mean and compute the 95% confidence interval

    :param ref: a dictionary with reference measure to test hypothesis sample match reference using t-student
    :param measures_list: a dictionary containing a list of measures for each type of measure
    :return: a dictionary containing the mean of all measures for each type of measure
    """
    measures = dict()
    for key, values in measures_list.items():
        measures[key] = np.mean(values)
        if ref and key in ref:
            measures['lower_ci_' + key], measures['higher_ci_' + key] = confidence(values, measures[key])
            test = st.ttest_1samp(values, ref[key]).pvalue > 0.05
            measures['test_' + key] = test

    return measures


def expected_sojourn_time(_lambda, mu, rho, theta):
    """
    Compute the theoretical mean sojourn time

    :param _lambda: arrival rate
    :param mu: service rate
    :param rho: arrival rate / service rate
    :param theta: setup rate
    :return: theoretical mean sojourn time
    """
    return 1 / mu / (1 - rho) + 1 / theta


def expected_p_setup(_lambda, rho, theta):
    """
    Compute the theoretical value of P_SETUP

    :param _lambda: arrival rate
    :param rho: arrival rate / service rate
    :param theta: setup rate
    :return: theoretical value of P_SETUP
    """
    return (1 - rho) / (theta / _lambda + 1)


def expected_p_off(_lambda, rho, theta):
    """
    Compute the theoretical value of P_OFF

    :param _lambda: arrival rate
    :param rho: arrival rate / service rate
    :param theta: setup rate
    :return: theoretical value of P_SETUP
    """
    return (1 - rho) / (1 + _lambda / theta)


def confidence(sample, mean=None):
    """
    Compute the confidence interval for the provided sample using t-student method
    :param sample: the sample
    :param mean: the mean of the sample
    :return: the lower bound and higher bound of the confidence interval
    """
    return st.t.interval(0.95, len(sample)-1, loc=mean or np.mean(sample), scale=st.sem(sample))