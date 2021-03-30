'''
Utilities for visualising P(AT)IPPET simulations.

----- Contact

Tom Kaplan (Music Cognition Lab, QMUL, UK)
t.m.kaplan@qmul.ac.uk
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, multivariate_normal
import seaborn as sns

def plot_pippet_error(model, figsize=(18,12), title=None):
    ''' Visualise PIPPET agent's trial with contours over time

    :param model: agent instance, after .run() call (PATIPPET)
    :param figsize: tuple for width/height of figure ((int,int), default=(18,12))
    '''
    from matplotlib.ticker import FormatStrFormatter
    sns.set(style="whitegrid")
    cs = sns.color_palette('husl', 3)

    fig, ax = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1, 4]})
    if title:
        fig.suptitle(title, y=1.05)

    ax[0].plot(model.ts, model.template)
    ax[0].set_ylabel('Expectation (λ)')
    ax[0].set_xticklabels([])
    
    std = 2*np.sqrt(model.V_s)
    ax[1].plot(model.ts, model.phibar_s, c=cs[0])
    ax[1].fill_between(model.ts, model.phibar_s-std, model.phibar_s+std, alpha=0.5, facecolor=cs[0])

    # Expected events
    for i in set(model.i_es): #- set(model.i_s):
        ax[1].axvline(model.ts[i], color=cs[1], alpha=0.85, linestyle=':', linewidth=2, label='Expected')
    # Actual events
    for i in set(model.i_s):
        ax[1].axvline(model.ts[i], color=cs[2], alpha=0.55, linestyle='-', linewidth=2, label='Events')

    ax[1].set_ylabel('Phase (Φ)')
    ax[1].set_xlabel('Time (s)')
    
    handles, labels = ax[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax[1].legend(by_label.values(), by_label.keys())

    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0.1)
    return fig

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
    cs_exp = sns.color_palette('cividis', n)

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
    from patippet import PIPPETParams, PATIPPETParams, PIPPET, PATIPPET

    #Example: PIPPET
    e_means = np.array([.25, .5, .75, 1.])
    e_vars = np.array([0.0001]).repeat(len(e_means))
    e_lambdas = np.array([0.01]).repeat(len(e_means))
    e_times = [1.]
    lambda_0 = 0.01
    dt = 0.001
    sigma_phase = 0.05
    overtime = 0.2

    kp = PIPPETParams(e_means, e_vars, e_lambdas, e_times, lambda_0,
                      dt, sigma_phase, overtime=overtime)
    kb = PIPPET(kp, drift=True)

    kb.prepare()
    kb.run()

    plot_pippet_error(kb, figsize=(6,3))

    #Example: PATIPPET
    '''
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
                        dt, sigma_phase, sigma_tempo=sigma_tempo, overtime=overtime)
    kb = PATIPPET(kp, drift=True)
    kb.prepare()
    kb.run()

    plot_patippet_contours(kb, n=10, figsize=(6,3), tminus1=True)
    '''


