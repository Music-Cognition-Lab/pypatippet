'''
Utilities for visualising PATIPPET simulations.

----- Contact

Tom Kaplan (Music Cognition Lab, QMUL, UK)
t.m.kaplan@qmul.ac.uk
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, multivariate_normal
import seaborn as sns

def plot_patippet_contours(model, n=5, figsize=(18,12), tminus1=True, max_y=2.0):
    ''' Visualise PATIPPET agent's trial with contours over time

    TODO: Everything. This is awful.
    - It should be passed an axis and extra plotting args/kwargs.
    - Separation of temporal expectation template into paired plot.
    - Continuous sufficient statistics over timesteps.

    :param model: agent instance, after .run() call (PATIPPET)
    :param n: number of contours to draw per event (int, default=5)
    :param figsize: tuple for width/height of figure ((int,int), default=(18,12))
    :param tminus1: whether to plot contours on timestep before event (bool, default=True)
    :param max_y: upper limit on y axis (float, default=2.0)
    '''
    sns.set(style="whitegrid")
    sns.set_palette(sns.color_palette('Set2', 1))
    cs_stim = sns.color_palette('viridis', n)
    cs_exp = sns.color_palette('Greys', n)

    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize=figsize)

    def _plot_ellipses(xbar, Sigma, X, Y, alpha, cols=cs_stim):
        g = multivariate_normal(mean=xbar, cov=Sigma)
        Z = g.pdf(np.dstack((X, Y)))
        axs[0].contour(X, Y, Z, np.arange(1, n), colors=cols, alpha=alpha)

    # Filtering
    X, Y = np.meshgrid(np.arange(-0.2, model.t_max+0.01, step=0.01),
                       np.arange(0.0, max_y+0.01, step=0.01))

    # Starting distribution
    _plot_ellipses(model.xbar_s[:, 0], model.Sigma_s[:, :, 0], X, Y, 0.8, cols=cs_exp)

    # Expected events
    for i in [i for i in model.i_es if i not in model.i_s]:
        _plot_ellipses(model.xbar_s[:, i], model.Sigma_s[:, :, i], X, Y, 0.8, cols=cs_exp)

    # On-event changes
    for i in model.i_s:
        if tminus1:
            _plot_ellipses(model.xbar_s[:, i-1], model.Sigma_s[:, :, i-1], X, Y, 0.7, cols=cs_stim)
        _plot_ellipses(model.xbar_s[:, i], model.Sigma_s[:, :, i], X, Y, 0.8, cols=cs_stim)

    # Template
    template = np.zeros(model.n_ts)
    for i in range(len(model.p.e_means)):
        pdf = norm.pdf(model.ts, loc=model.p.e_means[i], scale=(model.p.e_vars[i])**0.5)
        template += model.p.e_lambdas[i] * pdf
    template += model.p.lambda_0
    axs[1].plot(model.ts, template*.2)

    axs[0].set_xlim([-0.2, None])
    axs[1].set_xlim([-0.2, None])

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    from patippet import PATIPPETParams, PATIPPET

    e_means = np.array([.25, .5, .75, 1.])
    e_vars = np.array([0.001]).repeat(len(e_means))
    e_lambdas = np.array([0.02]).repeat(len(e_means))
    e_times = [1.]
    lambda_0 = 0.01
    dt = 0.001
    sigma_phase = 0.05
    sigma_tempo = 0.05
    overtime = 0.2

    kp = PATIPPETParams(e_means, e_vars, e_lambdas, e_times, lambda_0,
                        dt, sigma_phase, sigma_tempo, overtime=overtime)
    kb = PATIPPET(kp, drift=True)
    kb.prepare()
    kb.run()

    plot_patippet_contours(kb, n=10, figsize=(6,3), tminus1=True)
