'''
P(AT)IPPET: Phase (And Tempo) Inference from Point Process Event Timing [1]

[1] Cannon, Jonathan Joseph. "PIPPET: A Bayesian framework for generalized entrainment
to stochastic rhythms." bioRxiv (2020).

----- Information

See Jonathan Cannon's original MATLAB implementation:
    https://github.com/joncannon/PIPPET

----- TODO

- Tests. At least, re-produce from PIPPET pre-print [1]
- Multiple streams. This should be a simple change.
- Zenodo/DOI TBC...
- Parameter options for cyclic event sequences.

----- Contact

Tom Kaplan (Music Cognition Lab, QMUL, UK)
t.m.kaplan@qmul.ac.uk

'''
import attr
import numpy as np
from scipy.stats import norm

@attr.s
class PIPPETParams(object):
    ''' Parameter wrapper, struct-like '''
    e_means = attr.ib()
    e_vars = attr.ib()
    e_lambdas = attr.ib()
    e_times = attr.ib()
    lambda_0 = attr.ib()
    dt = attr.ib()
    sigma_phase = attr.ib()
    mean0 = attr.ib(default=0)
    var0 = attr.ib(default=0.0002)
    eta_phi = attr.ib(default=0.0)
    eta_e = attr.ib(default=0.0)
    overtime = attr.ib(default=0.0)
    t0 = attr.ib(default=0.0)

    def __attrs_post_init__(self):
        self.N = len(self.e_means)


class PIPPET(object):
    ''' Phase Inference from Point Process Event Timing '''

    def __init__(self, params, drift=True):
        self.p = params
        self._drift_between = drift
        self.prepare()

    def prepare(self, clear_state=True):
        ''' Initialise agent based on current parameter set '''

        # Perceived event times, add some noise
        self.e_times_p = self.p.e_times + np.random.randn(*self.p.e_times.shape) * self.p.eta_e

        self.t_max = max(self.e_times_p) + self.p.overtime
        self.ts = np.arange(self.p.t0, self.t_max+self.p.dt, step=self.p.dt)
        self.n_ts = self.ts.shape[0]

        # Sufficient statistics
        if clear_state:
            self.phibar_s = np.zeros(self.n_ts)
            self.phibar_s[0] = self.p.mean0
            self.V_s = np.zeros(self.n_ts)
            self.V_s[0] = self.p.var0

        # Current event index
        self.e_i = 0
        # Event onsets (within self.ts), once detected
        self.i_s = []
        # Expected event onsets (within self.ts)
        self.i_es = []

    @property
    def template(self):
        template = np.zeros(self.n_ts)
        for i in range(len(self.p.e_means)):
            pdf = norm.pdf(self.ts, loc=self.p.e_means[i], scale=(self.p.e_vars[i])**0.5)
            template += self.p.e_lambdas[i] * pdf
        template += self.p.lambda_0
        return template

    def is_onset(self, t_prev, t, i=0, stimulus=True):
        ''' Check whether event occurs between two timesteps

        :param t_prev: beginning of window, exclusive (float)
        :param t: end of window, inclusive (float)
        :param i: search from this index in event sequence (int, default=0)
        :param stimulus: check stimulus, else expected means (bool, defult=True)
        '''
        es = self.e_times_p if stimulus else self.p.e_means
        for e_i in range(i, len(es)):
            e_t = es[e_i]
            if t >= e_t and t_prev <= e_t:
                return True
        return False

    def phibar_i(self, phibar, V):
        return (phibar/V + self.p.e_means/self.p.e_vars)/(1/V + 1/self.p.e_vars)

    def K_i(self, V):
        return 1/(1/V + 1/self.p.e_vars)

    def lambda_i(self, phibar, V, o=0):
        gauss = norm.pdf(phibar, loc=self.p.e_means[o:], scale=(self.p.e_vars[o:] + V)**0.5)
        return self.p.e_lambdas[o:] * gauss

    def lambda_hat(self, phibar, V, o=0):
        return self.p.lambda_0 + np.sum(self.lambda_i(phibar, V, o=o))

    def phi_hat(self, phibar, V):
        phi_hat = self.p.lambda_0 * phibar
        phi_hat += np.sum(self.lambda_i(phibar, V) * self.phibar_i(phibar, V))
        return phi_hat / self.lambda_hat(phibar, V)

    def V_hat(self, phibar_curr, phibar_prev, V):
        V_hat = self.p.lambda_0 * (V + (phibar_prev-phibar_curr)**2)
        a = self.lambda_i(phibar_prev, V)
        b = self.K_i(V) + (self.phibar_i(phibar_prev, V)-phibar_curr)**2
        V_hat += np.sum(a * b)
        return V_hat / self.lambda_hat(phibar_prev, V)

    def step(self, i, phibar_prev=None, V_prev=None):
        phibar_prev = self.phibar_s[i-1] if not phibar_prev else phibar_prev
        V_prev = self.V_s[i-1] if not V_prev else V_prev

        # Tapping noise
        noise = np.sqrt(self.p.dt) * self.p.eta_phi * np.random.randn()

        # Drift between events
        dphibar = 0
        if self._drift_between:
            dphibar = self.lambda_hat(phibar_prev, V_prev)
            dphibar *= (self.phi_hat(phibar_prev, V_prev) - phibar_prev)
        phibar = phibar_prev + self.p.dt * (1 - dphibar) + noise

        dV = 0
        if self._drift_between:
            dV = self.lambda_hat(phibar_prev, V_prev)
            dV *= (self.V_hat(phibar, phibar_prev, V_prev) - V_prev)
        C = V_prev + self.p.dt * (self.p.sigma_phase**2 - dV)

        # If event, update accordingly
        t_prev, t = self.ts[i-1], self.ts[i]
        if self.is_onset(t_prev, t, self.e_i):
            phibar_new = self.phi_hat(phibar, C)
            C = self.V_hat(phibar_new, phibar, C)
            phibar = phibar_new
            self.e_i += 1
            self.i_s.append(i)

        # Note index of expected event (e.g. subdivision)
        if self.is_onset(t_prev, t, stimulus=False):
            self.i_es.append(i)

        # Update sufficient statistics and return
        self.phibar_s[i] = phibar
        self.V_s[i] = C

        return phibar, C

    def run(self):
        ''' Run PIPPET over timeline/events configured in parameter set '''
        for i in range(1, self.n_ts):
            self.step(i)


if __name__ == "__main__":
    import time

    #Example: PIPPET
    e_means = np.array([.25, .5, .75, 1.])
    e_vars = np.array([0.0001]).repeat(len(e_means))
    e_lambdas = np.array([0.02]).repeat(len(e_means))
    e_times = np.array([1.])
    lambda_0 = 0.01
    dt = 0.001
    sigma_phase = 0.05
    overtime = 0.2

    kp = PIPPETParams(e_means, e_vars, e_lambdas, e_times, lambda_0,
                      dt, sigma_phase, overtime=overtime)
    kb = PIPPET(kp, drift=True)

    kb.prepare()
    before = time.time()
    kb.run()
    print('Took {}s\n'.format(np.round(time.time() - before, 2)))

