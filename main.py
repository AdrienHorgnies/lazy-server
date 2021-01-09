"""
Orchestrate the simulations and display the result in a comprehensive way
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import SeedSequence, SFC64, Generator
from typing import List, Callable
from collections import defaultdict
from scipy.stats import ttest_1samp

from measures import compute_measures, append_measures, mean_measures, expected_sojourn_time, expected_p_off, \
    expected_p_setup
from simulations import exponential_simulation
from tqdm import tqdm

# the number of repetitions for each value of rho
N_SIM = 100
# the time at which the simulation stops
TAU = 1000
# the number of values tested for rho between 0.05 and 0.95
STEPS = 20
# the factor by which the small tau test is smaller
TAU_FACTOR = 10


def get_exponential_results(spawn_generators: Callable[[int], List[Generator]]):
    """
    Run all the required simulations for the exponential service time model and display results in CLI or GUI depending
    on the nature of the result.

    :param spawn_generators: a function able to spawn generators
    :return: None
    """
    progress_bar = tqdm(total=21 * STEPS * N_SIM // TAU_FACTOR)
    messages = []

    # first parameters set, lambda in [0.05, 0.95], mu = 1 and theta = 0.4
    rhos = np.linspace(0.05, 0.95, STEPS)
    mu = 1
    lambdas = mu * rhos
    theta = 0.4
    measures_by_rho = defaultdict(list)
    for _lambda in lambdas:
        measures_same_rho = defaultdict(list)

        ref = {
            "mean_sojourn": expected_sojourn_time(_lambda, mu, _lambda / mu, theta),
            "p_setup": expected_p_setup(_lambda, _lambda / mu, theta),
            "p_off": expected_p_off(_lambda, _lambda / mu, theta)
        }
        for _ in range(N_SIM):
            results = exponential_simulation(spawn_generators(3), _lambda, mu, theta, TAU)
            measures = compute_measures(*results)
            append_measures(measures_same_rho, measures)
            progress_bar.update(1)

        append_measures(measures_by_rho, mean_measures(measures_same_rho, ref))

    # Checking hypotheses, mu = 1
    messages.append('# Checking hypotheses, mu = 1')
    sojourn_test = sum(measures_by_rho['test_mean_sojourn']) / len(measures_by_rho['test_mean_sojourn'])
    messages.append('test for sojourn_time : %s' % sojourn_test)
    p_setup_test = sum(measures_by_rho['test_p_setup']) / len(measures_by_rho['test_p_setup'])
    messages.append('test for p_setup : %s' % p_setup_test)
    p_off_test = sum(measures_by_rho['test_p_off']) / len(measures_by_rho['test_p_off'])
    messages.append('test for p_off : %s' % p_off_test)

    # Graph for theoretical rho, measured rho and utilization
    actual_rhos = lambdas * np.array(measures_by_rho['mean_service'])

    fig_hyp, ax_rho = plt.subplots()
    fig_hyp.canvas.set_window_title('exponential-checking-hypotheses')

    ax_rho.set(xlabel=r'$\rho$ (expected)', ylabel='value', title=r'$\rho$ by its expected value ($\mu = 1$)')
    ax_rho.plot(rhos, rhos, label=r'expected $\rho$')
    ax_rho.step(rhos, actual_rhos, label=r'actual $\rho = \lambda \mathbb{E}[B]$')
    ax_rho.step(rhos, measures_by_rho['utilization'], label=fr'$\overline{"{x}"}$ ($\tau = {TAU}$)')

    # graphing results for mu = 1
    fig_sojourn, ax_sojourn = plt.subplots()
    fig_sojourn.canvas.set_window_title('exponential-sojourn')
    ax_sojourn.step(rhos, measures_by_rho['mean_sojourn'], label=r'measured ($\mu = 1$)')
    ax_sojourn.plot(rhos, expected_sojourn_time(lambdas, mu, rhos, theta), label=r'theoretical ($\mu = 1$)')
    ax_sojourn.fill_between(rhos, measures_by_rho['lower_ci_mean_sojourn'], measures_by_rho['higher_ci_mean_sojourn'],
                            alpha=0.4)
    ax_sojourn.set(xlabel=r'$\rho$', ylabel='duration', title=r'$\mathbb{E}[S]$ by $\rho$')

    fig_setup, ax_p_setup = plt.subplots()
    fig_setup.canvas.set_window_title('exponential-setup')
    ax_p_setup.step(rhos, measures_by_rho['p_setup'], label=r'measured ($\mu = 1$)')
    ax_p_setup.plot(rhos, expected_p_setup(lambdas, rhos, theta), label=r'theoretical ($\mu = 1$)')
    ax_p_setup.fill_between(rhos, measures_by_rho['lower_ci_p_setup'], measures_by_rho['higher_ci_p_setup'], alpha=0.4)
    ax_p_setup.set(xlabel=r'$\rho$', ylabel='$P_{SETUP}$', title=r'$P_{SETUP}$ by $\rho$')

    fig_off, ax_p_off = plt.subplots()
    fig_off.canvas.set_window_title('exponential-off')
    ax_p_off.step(rhos, measures_by_rho['p_off'], label=r'measured ($\mu = 1$)')
    ax_p_off.plot(rhos, expected_p_off(lambdas, rhos, theta), label=r'theoretical ($\mu = 1$)')
    ax_p_off.fill_between(rhos, measures_by_rho['lower_ci_p_off'], measures_by_rho['higher_ci_p_off'], alpha=0.4)
    ax_p_off.set(xlabel=r'$\rho$', ylabel='$P_{OFF}$', title=r'$P_{OFF}$ by $\rho$')

    # Same simulations but smaller TAU to prove utilization will diverge
    small_tau = TAU // TAU_FACTOR
    measures_by_rho = defaultdict(list)
    for _lambda in lambdas:
        measures_same_rho = defaultdict(list)

        for _ in range(N_SIM):
            results = exponential_simulation(spawn_generators(3), _lambda, mu, theta, small_tau)
            measures = compute_measures(*results)
            append_measures(measures_same_rho, measures)
            if _ % TAU_FACTOR == 0:
                progress_bar.update(1)

        append_measures(measures_by_rho, mean_measures(measures_same_rho))

    ax_rho.step(rhos, measures_by_rho['utilization'], label=fr'$\overline{"{x}"}$ ($\tau = {small_tau}$)')

    # second parameters set, lambda = 1, mu in [1/0.95, 1/0.05] and theta = 0.4
    _lambda = 1
    mus = _lambda / rhos
    theta = 0.4
    measures_by_rho = defaultdict(list)
    for mu in mus:
        measures_same_rho = defaultdict(list)

        ref = {
            "mean_sojourn": expected_sojourn_time(_lambda, mu, _lambda / mu, theta),
            "p_setup": expected_p_setup(_lambda, _lambda / mu, theta),
            "p_off": expected_p_off(_lambda, _lambda / mu, theta)
        }
        for _ in range(N_SIM):
            results = exponential_simulation(spawn_generators(3), _lambda, mu, theta, TAU)
            measures = compute_measures(*results)
            append_measures(measures_same_rho, measures)
            progress_bar.update(1)

        append_measures(measures_by_rho, mean_measures(measures_same_rho, ref))

    # Checking hypotheses, lambda = 1
    messages.append('# Checking hypotheses, lambda = 1')
    sojourn_test = sum(measures_by_rho['test_mean_sojourn']) / len(measures_by_rho['test_mean_sojourn'])
    messages.append('test for sojourn_time : %s' % sojourn_test)
    p_setup_test = sum(measures_by_rho['test_p_setup']) / len(measures_by_rho['test_p_setup'])
    messages.append('test for p_setup : %s' % p_setup_test)
    p_off_test = sum(measures_by_rho['test_p_off']) / len(measures_by_rho['test_p_off'])
    messages.append('test for p_off : %s' % p_off_test)

    # graphing results for lambda = 1
    ax_sojourn.step(rhos, measures_by_rho['mean_sojourn'], label=r'measured ($\lambda = 1$)')
    ax_sojourn.plot(rhos, expected_sojourn_time(_lambda, mus, rhos, theta), label=r'theoretical ($\lambda = 1$)')
    ax_sojourn.set(xlabel=r'$\rho$', ylabel='time', title=r'$\mathbb{E}[S]$ by $\rho$')
    ax_sojourn.fill_between(rhos, measures_by_rho['lower_ci_mean_sojourn'], measures_by_rho['higher_ci_mean_sojourn'],
                            alpha=0.4)

    ax_p_setup.step(rhos, measures_by_rho['p_setup'], label=r'measured ($\lambda = 1$)')
    ax_p_setup.plot(rhos, expected_p_setup(_lambda, rhos, theta), label=r'theoretical ($\lambda = 1$)')
    ax_p_setup.fill_between(rhos, measures_by_rho['lower_ci_p_setup'], measures_by_rho['higher_ci_p_setup'], alpha=0.4)
    ax_p_setup.set(xlabel=r'$\rho$', ylabel='$P_{SETUP}$', title=r'$P_{SETUP}$ by $\rho$')

    ax_p_off.step(rhos, measures_by_rho['p_off'], label=r'measured ($\lambda = 1$)')
    ax_p_off.plot(rhos, expected_p_off(_lambda, rhos, theta), label=r'theoretical ($\lambda = 1$)')
    ax_p_off.fill_between(rhos, measures_by_rho['lower_ci_p_off'], measures_by_rho['higher_ci_p_off'], alpha=0.4)
    ax_p_off.set(xlabel=r'$\rho$', ylabel='$P_{OFF}$', title=r'$P_{OFF}$ by $\rho$')

    for msg in messages:
        print(msg)

    ax_rho.legend(loc='best')
    ax_sojourn.legend(loc='best')
    ax_p_setup.legend(loc='best')
    ax_p_off.legend(loc='best')


def main():
    description = 'Produce the different results presented in the project report. '
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('seed', nargs='?', type=int, help='seed to initialize the random generator')
    args = parser.parse_args()

    if args.seed:
        seed_seq = SeedSequence(args.seed)
    else:
        seed_seq = SeedSequence()
    print('Seed : ', seed_seq.entropy)

    def spawn_generator(n):
        """
        Spawn n random generators

        Generators are insured independent if you spawn less than 2^64 of them and you pull less than 2^64 variates for
        each generators

        :return: a list of n generators
        """
        return [Generator(SFC64(stream)) for stream in seed_seq.spawn(n)]

    get_exponential_results(spawn_generator)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
