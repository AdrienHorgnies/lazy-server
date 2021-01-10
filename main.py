"""
Orchestrate the simulations and display the result in a comprehensive way
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import SeedSequence, SFC64, Generator
from typing import List, Callable
from collections import defaultdict
from figures import make_figure

from measures import compute_measures, append_measures, mean_measures, exp_sojourn_time, exp_p_off, \
    exp_p_setup, erlang_sojourn_time, erlang_p_setup, erlang_p_off
from simulations import exponential_simulation, erlang_simulation
from tqdm import tqdm

# the number of repetitions for each value of rho
N_SIM = 100
# the time at which the simulation stops
TAU = 2000
# the number of values tested for rho between 0.05 and 0.95
STEPS = 20
# the factor by which the small tau test is smaller
TAU_FACTOR = 10


def get_exponential_results(spawn_generators: Callable[[int], List[Generator]], progress_bar):
    """
    Run all the required simulations for the exponential service time model and display results in CLI or GUI depending
    on the nature of the result.

    :param spawn_generators: a function able to spawn generators
    :param progress_bar: a tqdm progress_bar to report progress
    :return: measures by rho for mu = 1
    """
    # first parameters set, lambda in [0.05, 0.95], mu = 1 and theta = 0.4
    rhos = np.linspace(0.05, 0.95, STEPS)
    mu = 1
    lambdas = mu * rhos
    theta = 0.4
    measures_by_rho_mu = defaultdict(list)
    for _lambda in lambdas:
        measures_same_rho = defaultdict(list)

        ref = {
            "mean_sojourn": exp_sojourn_time(_lambda, mu, _lambda / mu, theta),
            "p_setup": exp_p_setup(_lambda, _lambda / mu, theta),
            "p_off": exp_p_off(_lambda, _lambda / mu, theta)
        }
        for _ in range(N_SIM):
            results = exponential_simulation(spawn_generators(3), _lambda, mu, theta, TAU)
            measures = compute_measures(*results)
            append_measures(measures_same_rho, measures)
            progress_bar.update(1)

        append_measures(measures_by_rho_mu, mean_measures(measures_same_rho, ref))
    measures_by_rho_mu = {k: np.array(v) for k, v in measures_by_rho_mu.items()}

    # Same simulations but smaller TAU to prove utilization will diverge
    small_tau = TAU // TAU_FACTOR
    measures_by_rho_tau = defaultdict(list)
    for _lambda in lambdas:
        measures_same_rho = defaultdict(list)

        for _ in range(N_SIM):
            results = exponential_simulation(spawn_generators(3), _lambda, mu, theta, small_tau)
            measures = compute_measures(*results)
            append_measures(measures_same_rho, measures)
            if _ % TAU_FACTOR == 0:
                progress_bar.update(1)

        append_measures(measures_by_rho_tau, mean_measures(measures_same_rho))
    measures_by_rho_tau = {k: np.array(v) for k, v in measures_by_rho_tau.items()}

    # second parameters set, lambda = 1, mu in [1/0.95, 1/0.05] and theta = 0.4
    _lambda = 1
    mus = _lambda / rhos
    theta = 0.4
    measures_by_rho_lam = defaultdict(list)
    for mu in mus:
        measures_same_rho = defaultdict(list)

        ref = {
            "mean_sojourn": exp_sojourn_time(_lambda, mu, _lambda / mu, theta),
            "p_setup": exp_p_setup(_lambda, _lambda / mu, theta),
            "p_off": exp_p_off(_lambda, _lambda / mu, theta)
        }
        for _ in range(N_SIM):
            results = exponential_simulation(spawn_generators(3), _lambda, mu, theta, TAU)
            measures = compute_measures(*results)
            append_measures(measures_same_rho, measures)
            progress_bar.update(1)

        append_measures(measures_by_rho_lam, mean_measures(measures_same_rho, ref))
    measures_by_rho_lam = {k: np.array(v) for k, v in measures_by_rho_lam.items()}

    # Graph for theoretical rho, measured rho, utilization and small tau utilization
    actual_rhos = lambdas * np.array(measures_by_rho_mu['mean_service'])

    fig_hyp, ax_rho = plt.subplots()
    fig_hyp.canvas.set_window_title('exp-rho-rho-x')
    fig_hyp.set(label='exp-rho-rho-x')

    ax_rho.set(xlabel=r'$\rho$ (expected)', ylabel='value', title=r'$\rho$ by its expected value ($\mu = 1$)')
    ax_rho.plot(rhos, rhos, label=r'expected $\rho$')
    ax_rho.step(rhos, actual_rhos, label=r'actual $\rho = \lambda \mathbb{E}[B]$')
    ax_rho.step(rhos, measures_by_rho_mu['utilization'], label=fr'$\overline{"{x}"}$ ($\tau = {TAU}$)')
    ax_rho.step(rhos, measures_by_rho_tau['utilization'], label=fr'$\overline{"{x}"}$ ($\tau = {small_tau}$)')
    ax_rho.legend(loc='best')

    # graph sojourn time
    fig_sojourn, ax_sojourn = plt.subplots()
    fig_sojourn.canvas.set_window_title('exp-mean_sojourn')
    fig_sojourn.set(label='exp-mean_sojourn')

    ax_sojourn.scatter(rhos, measures_by_rho_mu['mean_sojourn'], label=r'measured ($\mu = 1$)')
    ax_sojourn.scatter(rhos, measures_by_rho_lam['mean_sojourn'], label=r'measured ($\lambda = 1$)')
    ax_sojourn.plot(rhos, exp_sojourn_time(lambdas, mu, rhos, theta), label=r'theoretical ($\mu = 1$)')
    ax_sojourn.plot(rhos, exp_sojourn_time(_lambda, mus, rhos, theta), label=r'theoretical ($\lambda = 1$)')
    ax_sojourn.set(xlabel=r'$\rho$', ylabel='time', title=r'$\mathbb{E}[S]$ by $\rho$')
    ax_sojourn.legend(loc='best')

    # graph sojourn H_0 and confidence interval, mu = 1
    fig_soj_mu_test, ax_soj_mu_test = plt.subplots()
    fig_soj_mu_test.canvas.set_window_title('exp-mean_sojourn-test-mu')
    fig_soj_mu_test.set(label='exp-mean_sojourn-test-mu')

    good = measures_by_rho_mu['test_mean_sojourn']
    bad = np.invert(good)

    ax_soj_mu_test.plot(rhos, exp_sojourn_time(lambdas, mu, rhos, theta), label=r'theoretical')
    ax_soj_mu_test.scatter(rhos[good], measures_by_rho_mu['mean_sojourn'][good], label='$H_0$')
    ax_soj_mu_test.scatter(rhos[bad], measures_by_rho_mu['mean_sojourn'][bad], label='$H_1$')
    ax_soj_mu_test.fill_between(rhos, measures_by_rho_mu['lower_ci_mean_sojourn'],
                                measures_by_rho_mu['higher_ci_mean_sojourn'], alpha=0.4, label=r'CI')
    ax_soj_mu_test.legend(loc='best')
    ax_soj_mu_test.set(xlabel=r'$\rho$', ylabel='time', title=r'$\mathbb{E}[S]$ by $\rho$ (tests for $\mu = 1$)')

    # graph sojourn H_0 and confidence interval, lambda = 1
    fig_soj_lam_test, ax_soj_lam_test = plt.subplots()
    fig_soj_lam_test.canvas.set_window_title('exp-mean_sojourn-test-lam')
    fig_soj_lam_test.set(label='exp-mean_sojourn-test-lam')

    good = measures_by_rho_lam['test_mean_sojourn']
    bad = np.invert(good)

    ax_soj_lam_test.plot(rhos, exp_sojourn_time(_lambda, mus, rhos, theta), label=r'theoretical')
    ax_soj_lam_test.scatter(rhos[good], measures_by_rho_lam['mean_sojourn'][good], label='$H_0$')
    ax_soj_lam_test.scatter(rhos[bad], measures_by_rho_lam['mean_sojourn'][bad], label='$H_1$')
    ax_soj_lam_test.fill_between(rhos, measures_by_rho_lam['lower_ci_mean_sojourn'],
                                 measures_by_rho_lam['higher_ci_mean_sojourn'], alpha=0.4, label='CI')
    ax_soj_lam_test.legend(loc='best')
    ax_soj_lam_test.set(xlabel=r'$\rho$', ylabel='time', title=r'$\mathbb{E}[S]$ by $\rho$ (tests for $\lambda = 1$)')

    # graphing p_setup
    fig_setup, ax_p_setup = plt.subplots()
    fig_setup.canvas.set_window_title('exp-p_setup')
    fig_setup.set(label='exp-p_setup')
    ax_p_setup.plot(rhos, exp_p_setup(lambdas, rhos, theta), label=r'theoretical ($\mu = 1$)')
    ax_p_setup.plot(rhos, exp_p_setup(_lambda, rhos, theta), label=r'theoretical ($\lambda = 1$)')
    ax_p_setup.scatter(rhos, measures_by_rho_mu['p_setup'], label=r'measured ($\mu = 1$)')
    ax_p_setup.scatter(rhos, measures_by_rho_lam['p_setup'], label=r'measured ($\lambda = 1$)')
    ax_p_setup.legend(loc='best')
    ax_p_setup.set(xlabel=r'$\rho$', ylabel='probability', title=r'$P_{SETUP}$ by $\rho$')

    # graph p_setup test mu = 1
    fig_setup_test_mu, ax_p_setup_test_mu = plt.subplots()
    fig_setup_test_mu.canvas.set_window_title('exp-p_setup-test-mu')
    fig_setup_test_mu.set(label='exp-p_setup-test-mu')

    good = measures_by_rho_mu['test_p_setup']
    bad = np.invert(good)

    ax_p_setup_test_mu.plot(rhos, exp_p_setup(lambdas, rhos, theta), label=r'theoretical')
    ax_p_setup_test_mu.scatter(rhos[good], measures_by_rho_mu['p_setup'][good], label=r'$H_0$')
    ax_p_setup_test_mu.scatter(rhos[bad], measures_by_rho_mu['p_setup'][bad], label=r'$H_1$')
    ax_p_setup_test_mu.fill_between(rhos, measures_by_rho_mu['lower_ci_p_setup'],
                                    measures_by_rho_mu['higher_ci_p_setup'],
                                    alpha=0.4, label='CI')
    ax_p_setup_test_mu.set(xlabel=r'$\rho$', ylabel='probability', title=r'$P_{SETUP}$ by $\rho$ (tests for $\mu = 1$)')
    ax_p_setup_test_mu.legend(loc='best')

    # graph p_setup test lam = 1
    fig_setup_test_lam, ax_p_setup_test_lam = plt.subplots()
    fig_setup_test_lam.canvas.set_window_title('exp-p_setup-test-lam')
    fig_setup_test_lam.set(label='exp-p_setup-test-lam')

    good = measures_by_rho_lam['test_p_setup']
    bad = np.invert(good)

    ax_p_setup_test_lam.plot(rhos, exp_p_setup(_lambda, rhos, theta), label=r'theoretical')
    ax_p_setup_test_lam.scatter(rhos[good], measures_by_rho_lam['p_setup'][good], label=r'$H_0$')
    ax_p_setup_test_lam.scatter(rhos[bad], measures_by_rho_lam['p_setup'][bad], label=r'$H_1$')
    ax_p_setup_test_lam.fill_between(rhos, measures_by_rho_lam['lower_ci_p_setup'],
                                     measures_by_rho_lam['higher_ci_p_setup'],
                                     alpha=0.4, label='CI')
    ax_p_setup_test_lam.set(xlabel=r'$\rho$', ylabel='probability',
                            title=r'$P_{SETUP}$ by $\rho$ (tests for $\lambda = 1$)')
    ax_p_setup_test_lam.legend(loc='best')

    # graphing p_off
    fig_off, ax_p_off = plt.subplots()
    fig_off.canvas.set_window_title('exp-p_off')
    fig_off.set(label='exp-p_off')
    ax_p_off.plot(rhos, exp_p_off(lambdas, rhos, theta), label=r'theoretical ($\mu = 1$)')
    ax_p_off.plot(rhos, exp_p_off(_lambda, rhos, theta), label=r'theoretical ($\lambda = 1$)')
    ax_p_off.scatter(rhos, measures_by_rho_mu['p_off'], label=r'measured ($\mu = 1$)')
    ax_p_off.scatter(rhos, measures_by_rho_lam['p_off'], label=r'measured ($\lambda = 1$)')
    ax_p_off.legend(loc='best')
    ax_p_off.set(xlabel=r'$\rho$', ylabel='probability', title=r'$P_{OFF}$ by $\rho$')

    # graph p_off test mu = 1
    fig_off_test_mu, ax_p_off_test_mu = plt.subplots()
    fig_off_test_mu.canvas.set_window_title('exp-p_off-test-mu')
    fig_off_test_mu.set(label='exp-p_off-test-mu')

    good = measures_by_rho_mu['test_p_off']
    bad = np.invert(good)

    ax_p_off_test_mu.plot(rhos, exp_p_off(lambdas, rhos, theta), label=r'theoretical')
    ax_p_off_test_mu.scatter(rhos[good], measures_by_rho_mu['p_off'][good], label=r'$H_0$')
    ax_p_off_test_mu.scatter(rhos[bad], measures_by_rho_mu['p_off'][bad], label=r'$H_1$')
    ax_p_off_test_mu.fill_between(rhos, measures_by_rho_mu['lower_ci_p_off'],
                                  measures_by_rho_mu['higher_ci_p_off'],
                                  alpha=0.4, label='CI')
    ax_p_off_test_mu.set(xlabel=r'$\rho$', ylabel='probability', title=r'$P_{OFF}$ by $\rho$ (tests for $\mu = 1$)')
    ax_p_off_test_mu.legend(loc='best')

    # graph p_off test lam = 1
    fig_off_test_lam, ax_p_off_test_lam = plt.subplots()
    fig_off_test_lam.canvas.set_window_title('exp-p_off-test-lam')
    fig_off_test_lam.set(label='exp-p_off-test-lam')

    good = measures_by_rho_lam['test_p_off']
    bad = np.invert(good)

    ax_p_off_test_lam.plot(rhos, exp_p_off(_lambda, rhos, theta), label=r'theoretical')
    ax_p_off_test_lam.scatter(rhos[good], measures_by_rho_lam['p_off'][good], label=r'$H_0$')
    ax_p_off_test_lam.scatter(rhos[bad], measures_by_rho_lam['p_off'][bad], label=r'$H_1$')
    ax_p_off_test_lam.fill_between(rhos, measures_by_rho_lam['lower_ci_p_off'],
                                   measures_by_rho_lam['higher_ci_p_off'],
                                   alpha=0.4, label='CI')
    ax_p_off_test_lam.set(xlabel=r'$\rho$', ylabel='probability',
                          title=r'$P_{OFF}$ by $\rho$ (tests for $\lambda = 1$)')
    ax_p_off_test_lam.legend(loc='best')

    return measures_by_rho_mu


def get_erlang_results(spawn_generators: Callable[[int], List[Generator]], progress_bar):
    """
    Run all the required simulations for the exponential service time model and display results in CLI or GUI depending
    on the nature of the result.

    :param spawn_generators: a function able to spawn generators
    :param progress_bar: a tqdm progress_bar to report progress
    :return: measures by rho for mu = 1
    """
    rhos = np.linspace(0.05, 0.95, STEPS)
    n = 10
    b = 0.1
    theta = 0.2

    measures_by_rho = defaultdict(list)
    for rho in rhos:
        _lambda = rho / (n * b)
        measures_same_rho = defaultdict(list)

        ref = {
            "mean_sojourn": erlang_sojourn_time(_lambda, n, b, rho, theta),
            "p_setup": erlang_p_setup(_lambda, n, b, rho, theta),
            "p_off": erlang_p_off(_lambda, n, b, rho, theta)
        }
        for _ in range(N_SIM):
            results = erlang_simulation(spawn_generators(3), _lambda, n, b, theta, TAU)
            measures = compute_measures(*results)
            append_measures(measures_same_rho, measures)
            progress_bar.update(1)

        append_measures(measures_by_rho, mean_measures(measures_same_rho, ref))
    measures_by_rho = {k: np.array(v) for k, v in measures_by_rho.items()}

    make_figure(rhos, measures_by_rho, 'erlang', 'mean_sojourn', r'\mathbb{E}[S]', 'time')
    make_figure(rhos, measures_by_rho, 'erlang', 'p_setup', 'P_{SETUP}', 'probability')
    make_figure(rhos, measures_by_rho, 'erlang', 'p_off', 'P_{OFF}', 'probability')


def main():
    description = 'Produce the different results presented in the project report. '
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('seed', nargs='?', type=int, help='seed to initialize the random generator')
    parser.add_argument('--save-figures', dest='save', action='store_true',
                        help='Whether to save figures or not in ./out')
    args = parser.parse_args()

    if args.seed:
        seed_seq = SeedSequence(args.seed)
    else:
        seed_seq = SeedSequence()
    print('Seed : ', seed_seq.entropy)

    def spawn_generators(n):
        """
        Spawn n random generators

        Generators are insured independent if you spawn less than 2^64 of them and you pull less than 2^64 variates for
        each generators

        :return: a list of n generators
        """
        return [Generator(SFC64(stream)) for stream in seed_seq.spawn(n)]

    progress_bar = tqdm(total=31 * STEPS * N_SIM // TAU_FACTOR)

    # get_exponential_results(spawn_generators, progress_bar)
    get_erlang_results(spawn_generators, progress_bar)

    plt.tight_layout()
    if args.save:
        for label in plt.get_figlabels():
            plt.figure(label).savefig('./out/' + label + '.png')
        with open('./out/seed.txt', 'w+') as f:
            f.write(str(seed_seq.entropy) + '\n')
    plt.show()


if __name__ == '__main__':
    main()
