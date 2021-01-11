import matplotlib.pyplot as plt
import numpy as np


def make_figure(rhos, data, prefix, measure_name, measure_symbol, unit_name, exp):
    fig, ax = plt.subplots()
    fig_name = f'{prefix}-{measure_name}'
    fig.canvas.set_window_title(fig_name)
    fig.set(label=fig_name)

    ax.set(xlabel=r'$\rho$', ylabel=unit_name, title=fr'${measure_symbol}$ by $\rho$')
    ax.plot(rhos, data['theo_' + measure_name], label='theoretical (Erlang)')
    ax.scatter(rhos, data[measure_name], label='measured (Erlang)')
    ax.scatter(rhos, exp[measure_name], label='measured (Exponential)')
    ax.legend(loc='best')

    fig_test, ax_test = plt.subplots()
    fig_name = f'{prefix}-{measure_name}-test'
    fig_test.canvas.set_window_title(fig_name)
    fig_test.set(label=fig_name)

    good = data['test_' + measure_name]
    bad = np.invert(good)

    ax_test.set(xlabel=r'$\rho$', ylabel=unit_name, title=fr'${measure_symbol}$ by $\rho$')
    ax_test.plot(rhos, data['theo_' + measure_name], label='theoretical')
    ax_test.scatter(rhos[good], data[measure_name][good], label='$H_0$')
    ax_test.scatter(rhos[bad], data[measure_name][bad], label='$H_1$')
    ax_test.fill_between(rhos, data['lower_ci_' + measure_name],
                         data['higher_ci_' + measure_name], alpha=0.4, label='CI')
    ax_test.legend(loc='best')
