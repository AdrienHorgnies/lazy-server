import matplotlib.pyplot as plt


def make_figure(rhos, data, prefix, measure_name, measure_symbol, unit_name):
    fig, ax = plt.subplots()
    fig_name = f'{prefix}-{measure_name}'
    fig.canvas.set_window_title(fig_name)
    fig.set(label=fig_name)

    ax.set(xlabel=r'$\rho$', ylabel=unit_name, title=fr'${measure_symbol}$ by $\rho$')
    ax.plot(rhos, data['theo_' + measure_name], label='theoretical')
    ax.scatter(rhos, data[measure_name], label='measured')
