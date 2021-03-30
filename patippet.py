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

    def _is_onset(self, t_prev, t, i=0, stimulus=True):
        ''' Check whether event occurs between two timesteps

        :param t_prev: beginning of window, exclusive (float)
        :param t: end of window, inclusive (float)
        :param i: search from this index in event sequence (int, default=0)
        :param stimulus: check stimulus, else expected means (bool, defult=True)
        '''
        es = self.e_times_p if stimulus else self.p.e_means
        for e_i in range(i, len(es)):
            e_t = es[e_i]
            if t >= e_t and t_prev < e_t:
                return True
        return False

    def _phibar_i(self, phibar, V):
        return (phibar/V + self.p.e_means/self.p.e_vars)/(1/V + 1/self.p.e_vars)

    def _K_i(self, V):
        return 1/(1/V + 1/self.p.e_vars)

    def _lambda_i(self, phibar, V):
        gauss = norm.pdf(phibar, loc=self.p.e_means, scale=(self.p.e_vars + V)**0.5)
        return self.p.e_lambdas * gauss

    def _lambda_hat(self, phibar, V):
        return self.p.lambda_0 + np.sum(self._lambda_i(phibar, V))

    def _phi_hat(self, phibar, V):
        phi_hat = self.p.lambda_0 * phibar
        phi_hat += np.sum(self._lambda_i(phibar, V) * self._phibar_i(phibar, V))
        return phi_hat / self._lambda_hat(phibar, V)

    def _V_hat(self, phibar_curr, phibar_prev, V):
        V_hat = self.p.lambda_0 * (V + (phibar_prev-phibar_curr)**2)
        a = self._lambda_i(phibar_prev, V)
        b = self._K_i(V) + (self._phibar_i(phibar_prev, V)-phibar_curr)**2
        V_hat += np.sum(a * b)
        return V_hat / self._lambda_hat(phibar_prev, V)

    def run(self):
        ''' Run PIPPET over timeline/events configured in parameter set '''
        e_i = 0
        for i in range(1, self.n_ts):
            phibar_prev = self.phibar_s[i-1]
            V_prev = self.V_s[i-1]

            # Tapping noise
            noise = np.sqrt(self.p.dt) * self.p.eta_phi * np.random.randn()

            # Drift between events
            dphibar = 0
            if self._drift_between:
                dphibar = self._lambda_hat(phibar_prev, V_prev)
                dphibar *= (self._phi_hat(phibar_prev, V_prev) - phibar_prev)
            phibar = phibar_prev + self.p.dt * (1 - dphibar) + noise

            dV = 0
            if self._drift_between:
                dV = self._lambda_hat(phibar_prev, V_prev)
                dV *= (self._V_hat(phibar, phibar_prev, V_prev) - V_prev)
            C = V_prev + self.p.dt * (self.p.sigma_phase**2 - dV)

            # If event, update accordingly
            t_prev, t = self.ts[i-1], self.ts[i]
            if self._is_onset(t_prev, t, e_i):
                phibar_new = self._phi_hat(phibar, C)
                C = self._V_hat(phibar_new, phibar, C)
                phibar = phibar_new
                e_i += 1
                self.i_s.append(i)

            # Note index of expected event (e.g. subdivision)
            if self._is_onset(t_prev, t, stimulus=False):
                self.i_es.append(i)

            self.V_s[i] = C
            self.phibar_s[ i] = phibar


@attr.s
class PATIPPETParams(PIPPETParams):
    sigma_tempo = attr.ib(default=0.0)
    mean0 = attr.ib(default=[0,1])
    var0 = attr.ib(default=np.array([[0.0001, 0], [0, 0.04]]))

class PATIPPET(PIPPET):
    ''' Phase (And Tempo) Inference from Point Process Event Timing '''

    def prepare(self, clear_state=True):
        ''' Initialise agent based on current parameter set '''

        # Most of the initialisation is the same as PIPPET, other than 2D suff. stats
        super().prepare(clear_state=False)

        if clear_state:
            self.xbar_s = np.zeros((2, self.n_ts))
            self.xbar_s[:,0] = self.p.mean0
            self.Sigma_s = np.zeros((2, 2, self.n_ts))
            self.Sigma_s[:,:,0] = self.p.var0

    def _lambda_i(self, xbar, Sigma, i):
        gauss = norm.pdf(xbar[0], loc=self.p.e_means[i], scale=(self.p.e_vars[i] + Sigma[0,0])**0.5)
        return self.p.e_lambdas[i] * gauss

    def _lambda_hat(self, xbar, Sigma):
        lam_hat = self.p.lambda_0*xbar[1]
        for i in range(self.p.e_means.shape[0]):
            lam_hat += self._lambda_i(xbar, Sigma, i)*self._xbar_i(xbar, Sigma, i)
        return lam_hat

    def _xbar_i(self, xbar, Sigma, i):
        a = np.linalg.inv(Sigma) + np.array([[1/self.p.e_vars[i],0], [0,0]])
        b = np.linalg.solve(Sigma, xbar) + np.array([self.p.e_means[i]/self.p.e_vars[i], 0])
        return np.linalg.solve(a, b)

    def _x_hat(self, xbar, Sigma):
        x_sum = self.p.lambda_0 * (Sigma[:,1] + xbar*xbar[1])
        for i in range(self.p.e_means.shape[0]):
            K_i = self._K_i(Sigma, i)[:,1]
            xbar_i = self._xbar_i(xbar, Sigma, i)
            x_sum += self._lambda_i(xbar, Sigma, i) * (K_i + xbar_i*xbar_i[1])
        return x_sum/self._lambda_hat(xbar, Sigma)

    def _K_i(self, Sigma, i):
        return np.linalg.inv(np.linalg.inv(Sigma) + np.array([[1./self.p.e_vars[i],0], [0,0]]))

    def _Sigma_hat(self, xbar_curr, xbar_prev, Sigma):
        xbar_diff = xbar_prev - xbar_curr
        xbar_diff_T = xbar_diff.conj().T

        Sigma_sum = xbar_prev[1] * (Sigma + xbar_diff*xbar_diff_T)
        Sigma_sum += xbar_diff*Sigma[1,:] + Sigma[:,1]*xbar_diff_T
        Sigma_sum *= self.p.lambda_0

        for i in range(self.p.e_means.shape[0]):
            K_i = self._K_i(Sigma, i)
            xbar_i = self._xbar_i(xbar_prev, Sigma, i)
            xbar_i_diff = xbar_i - xbar_curr
            xbar_i_diff_T = xbar_i_diff.conj().T

            scalar = xbar_i[1]*(K_i + xbar_i_diff*xbar_diff_T)
            scalar += xbar_diff*K_i[1,:] + K_i[:,1]*xbar_diff_T

            Sigma_sum += self._lambda_i(xbar_prev, Sigma, i)
            Sigma_sum *= scalar

        return Sigma_sum/self._lambda_hat(xbar_prev, Sigma)

    def _between_events(self, xbar_prev, Sigma_prev, drift=True):
        ''' Drift of sufficient statistics between stimulus events '''

        # Update mu
        dxbar_m = np.array([xbar_prev[1], 0])
        dxbar = 0.
        if drift:
            dxbar = self._x_hat(xbar_prev, Sigma_prev) - xbar_prev
            dxbar = self._lambda_hat(xbar_prev, Sigma_prev) * dxbar
        dxbar = self.p.dt * (dxbar_m - dxbar)
        xbar = xbar_prev + dxbar

        # Update Sigma
        Sigma_m = np.array([[self.p.sigma_phase**2 + 2*Sigma_prev[0,1], Sigma_prev[1,1]],
                            [Sigma_prev[1,1], self.p.sigma_tempo**2]])
        dSigma = 0.
        if drift:
            dSigma = self._Sigma_hat(xbar, xbar_prev, Sigma_prev) - Sigma_prev
            dSigma = self._lambda_hat(xbar_prev, Sigma_prev) * dSigma
        dSigma = self.p.dt * (Sigma_m - dSigma)
        Sigma = Sigma_prev + dSigma

        return xbar, Sigma

    def _on_event(self, xbar, Sigma):
        ''' Jump of sufficient statistics on event '''
        xbar_next = self._x_hat(xbar, Sigma)
        Sigma_next = self._Sigma_hat(xbar_next, xbar, Sigma)
        return xbar_next, Sigma_next

    def run(self):
        ''' Run PATIPPET over timeline/events configured in parameter set '''
        e_i = 0
        for i in range(1, self.n_ts):
            xbar_prev = self.xbar_s[:, i-1]
            Sigma_prev = self.Sigma_s[:, :, i-1]

            # Update for time step
            xbar, Sigma = self._between_events(xbar_prev, Sigma_prev, self._drift_between)

            # If event, update accordingly
            t_prev, t = self.ts[i-1], self.ts[i]
            if self._is_onset(t_prev, t, e_i):
                xbar, Sigma = self._on_event(xbar, Sigma)
                e_i += 1
                self.i_s.append(i)

            # Note index of expected event (e.g. subdivision)
            if self._is_onset(t_prev, t, stimulus=False):
                self.i_es.append(i)

            self.Sigma_s[:, :, i] = Sigma
            self.xbar_s[:, i] = xbar

if __name__ == "__main__":
    import time

    '''
    #Example: PATIPPET

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
    '''

    #Example: PIPPET
    e_means = np.array([.25, .5, .75, 1.])
    e_vars = np.array([0.0001]).repeat(len(e_means))
    e_lambdas = np.array([0.02]).repeat(len(e_means))
    e_times = [1.]
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

