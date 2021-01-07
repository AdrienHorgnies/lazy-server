"""
Orchestrate the simulations and display the result in a comprehensive way
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import SeedSequence, SFC64, Generator
from typing import List, Callable
from collections import defaultdict

from measures import compute_measures, append_measures, mean_measures, mean_sojourn_time, compute_p_off, compute_p_setup
from simulations import exponential_simulation
from tqdm import tqdm


# the number of repetitions for each value of rho
N_SIM = 100
# the time at which the simulation stops
TAU = 100
# the number of values tested for rho between 0.05 and 0.95
STEPS = 100


def get_exponential_results(spawn_generators: Callable[[int], List[Generator]]):
    """
    Run all the required simulations for the exponential service time model and display results in CLI or GUI depending
    on the nature of the result.

    :param spawn_generators: a function able to spawn generators
    :return: None
    """
    progress_bar = tqdm(total=2 * STEPS * N_SIM)

    # first parameters set, lambda in [0.05, 0.95], mu = 1 and theta = 0.2
    rhos = np.linspace(0.05, 0.95, STEPS)
    mu = 1
    lambdas = mu * rhos
    theta = 0.2
    measures_by_rho = defaultdict(list)
    for _lambda in lambdas:
        measures_same_rho = defaultdict(list)

        for _ in range(N_SIM):
            results = exponential_simulation(spawn_generators(3), _lambda, mu, theta, TAU)
            measures = compute_measures(*results)
            append_measures(measures_same_rho, measures)
            progress_bar.update(1)

        append_measures(measures_by_rho, mean_measures(measures_same_rho))

    # Checking hypotheses
    actual_rhos = lambdas * np.array(measures_by_rho['service_mean'])

    fig_hyp, ax_rho = plt.subplots()
    fig_hyp.canvas.set_window_title('exponential-checking-hypotheses')

    ax_rho.set(xlabel=r'$\rho$ (expected)', ylabel='value', title=r'$\rho$ by its expected value ($\mu = 1$)')
    ax_rho.plot(rhos, rhos, label=r'expected $\rho$')
    ax_rho.step(rhos, actual_rhos, label=r'actual $\rho = \lambda \mathbb{E}[B]$')
    ax_rho.step(rhos, measures_by_rho['utilization'], label=r'$\overline{x}$')

    # expected results
    fig, (ax_sojourn, ax_p_setup, ax_p_off) = plt.subplots(ncols=3, sharex='all')
    fig.suptitle('Results for the exponential system', fontsize=16)
    fig.canvas.set_window_title('exponential-lazy-server-results')

    ax_sojourn.step(rhos, measures_by_rho['sojourn_mean'], label=r'measured ($\mu = 1$)')
    ax_sojourn.plot(rhos, mean_sojourn_time(lambdas, mu, rhos, theta), label=r'theoretical ($\mu = 1$)')
    ax_sojourn.set(xlabel=r'$\rho$', ylabel='duration', title=r'$\mathbb{E}[S]$ by $\rho$')
    ax_p_setup.step(rhos, measures_by_rho['p_setup'], label=r'measured ($\mu = 1$)')
    ax_p_setup.plot(rhos, compute_p_setup(lambdas, rhos, theta), label=r'theoretical ($\mu = 1$)')
    ax_p_setup.set(xlabel=r'$\rho$', ylabel='$P_{SETUP}$', title=r'$P_{SETUP}$ by $\rho$')
    ax_p_off.step(rhos, measures_by_rho['p_off'], label=r'measured ($\mu = 1$)')
    ax_p_off.plot(rhos, compute_p_off(lambdas, rhos, theta), label=r'theoretical ($\mu = 1$)')
    ax_p_off.set(xlabel=r'$\rho$', ylabel='$P_{OFF}$', title=r'$P_{OFF}$ by $\rho$')

    del lambdas, mu, measures_by_rho, measures_same_rho  # to avoid using values in next scenario

    # second parameters set, lambda = 1, mu in [1/0.95, 1/0.05] and theta = 0.2

    _lambda = 1
    mus = _lambda / rhos
    theta = 0.2
    measures_by_rho = defaultdict(list)
    for mu in mus:
        measures_same_rho = defaultdict(list)

        for _ in range(N_SIM):
            results = exponential_simulation(spawn_generators(3), _lambda, mu, theta, TAU)
            measures = compute_measures(*results)
            append_measures(measures_same_rho, measures)
            progress_bar.update(1)

        append_measures(measures_by_rho, mean_measures(measures_same_rho))

    ax_sojourn.step(rhos, measures_by_rho['sojourn_mean'], label=r'measured ($\lambda = 1$)')
    ax_sojourn.plot(rhos, mean_sojourn_time(_lambda, mus, rhos, theta), label=r'theoretical ($\lambda = 1$)')
    ax_sojourn.set(xlabel=r'$\rho$', ylabel='time', title=r'$\mathbb{E}[S]$ by $\rho$')
    ax_p_setup.step(rhos, measures_by_rho['p_setup'], label=r'measured ($\lambda = 1$)')
    ax_p_setup.plot(rhos, compute_p_setup(_lambda, rhos, theta), label=r'theoretical ($\lambda = 1$)')
    ax_p_setup.set(xlabel=r'$\rho$', ylabel='$P_{SETUP}$', title=r'$P_{SETUP}$ by $\rho$')
    ax_p_off.step(rhos, measures_by_rho['p_off'], label=r'measured ($\lambda = 1$)')
    ax_p_off.plot(rhos, compute_p_off(_lambda, rhos, theta), label=r'theoretical ($\lambda = 1$)')
    ax_p_off.set(xlabel=r'$\rho$', ylabel='$P_{OFF}$', title=r'$P_{OFF}$ by $\rho$')

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
