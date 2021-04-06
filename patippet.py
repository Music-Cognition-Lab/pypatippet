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

from pippet import PIPPETParams, PIPPET

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

    def lambda_i(self, xbar, Sigma, i):
        gauss = norm.pdf(xbar[0], loc=self.p.e_means[i], scale=(self.p.e_vars[i] + Sigma[0,0])**0.5)
        return self.p.e_lambdas[i] * gauss

    def lambda_hat(self, xbar, Sigma):
        lam_hat = self.p.lambda_0*xbar[1]
        for i in range(self.p.e_means.shape[0]):
            lam_hat += self.lambda_i(xbar, Sigma, i)*self.xbar_i(xbar, Sigma, i)
        return lam_hat

    def xbar_i(self, xbar, Sigma, i):
        a = np.linalg.inv(Sigma) + np.array([[1/self.p.e_vars[i],0], [0,0]])
        b = np.linalg.solve(Sigma, xbar) + np.array([self.p.e_means[i]/self.p.e_vars[i], 0])
        return np.linalg.solve(a, b)

    def x_hat(self, xbar, Sigma):
        x_sum = self.p.lambda_0 * (Sigma[:,1] + xbar*xbar[1])
        for i in range(self.p.e_means.shape[0]):
            K_i = self.K_i(Sigma, i)[:,1]
            xbar_i = self.xbar_i(xbar, Sigma, i)
            x_sum += self.lambda_i(xbar, Sigma, i) * (K_i + xbar_i*xbar_i[1])
        return x_sum/self.lambda_hat(xbar, Sigma)

    def K_i(self, Sigma, i):
        return np.linalg.inv(np.linalg.inv(Sigma) + np.array([[1./self.p.e_vars[i],0], [0,0]]))

    def Sigma_hat(self, xbar_curr, xbar_prev, Sigma):
        xbar_diff = xbar_prev - xbar_curr
        xbar_diff_T = xbar_diff.conj().T

        Sigma_sum = xbar_prev[1] * (Sigma + xbar_diff*xbar_diff_T)
        Sigma_sum += xbar_diff*Sigma[1,:] + Sigma[:,1]*xbar_diff_T
        Sigma_sum *= self.p.lambda_0

        for i in range(self.p.e_means.shape[0]):
            K_i = self.K_i(Sigma, i)
            xbar_i = self.xbar_i(xbar_prev, Sigma, i)
            xbar_i_diff = xbar_i - xbar_curr
            xbar_i_diff_T = xbar_i_diff.conj().T

            scalar = xbar_i[1]*(K_i + xbar_i_diff*xbar_diff_T)
            scalar += xbar_diff*K_i[1,:] + K_i[:,1]*xbar_diff_T

            Sigma_sum += self.lambda_i(xbar_prev, Sigma, i)
            Sigma_sum *= scalar

        return Sigma_sum/self.lambda_hat(xbar_prev, Sigma)

    def _between_events(self, xbar_prev, Sigma_prev, drift=True):
        ''' Drift of sufficient statistics between stimulus events '''

        # Update mu
        dxbar_m = np.array([xbar_prev[1], 0])
        dxbar = 0.
        if drift:
            dxbar = self.x_hat(xbar_prev, Sigma_prev) - xbar_prev
            dxbar = self.lambda_hat(xbar_prev, Sigma_prev) * dxbar
        dxbar = self.p.dt * (dxbar_m - dxbar)
        xbar = xbar_prev + dxbar

        # Update Sigma
        Sigma_m = np.array([[self.p.sigma_phase**2 + 2*Sigma_prev[0,1], Sigma_prev[1,1]],
                            [Sigma_prev[1,1], self.p.sigma_tempo**2]])
        dSigma = 0.
        if drift:
            dSigma = self.Sigma_hat(xbar, xbar_prev, Sigma_prev) - Sigma_prev
            dSigma = self.lambda_hat(xbar_prev, Sigma_prev) * dSigma
        dSigma = self.p.dt * (Sigma_m - dSigma)
        Sigma = Sigma_prev + dSigma

        return xbar, Sigma

    def _on_event(self, xbar, Sigma):
        ''' Jump of sufficient statistics on event '''
        xbar_next = self.x_hat(xbar, Sigma)
        Sigma_next = self.Sigma_hat(xbar_next, xbar, Sigma)
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
            if self.is_onset(t_prev, t, e_i):
                xbar, Sigma = self._on_event(xbar, Sigma)
                e_i += 1
                self.i_s.append(i)

            # Note index of expected event (e.g. subdivision)
            if self.is_onset(t_prev, t, stimulus=False):
                self.i_es.append(i)

            self.Sigma_s[:, :, i] = Sigma
            self.xbar_s[:, i] = xbar

if __name__ == "__main__":
    import time

    #Example: PATIPPET

    e_means = np.array([.25, .5, .75, 1.])
    e_vars = np.array([0.001]).repeat(len(e_means))
    e_lambdas = np.array([0.02]).repeat(len(e_means))
    e_times = np.array([1.])
    lambda_0 = 0.01
    dt = 0.001
    sigma_phase = 0.05
    sigma_tempo = 0.05
    overtime = 0.2

    kp = PATIPPETParams(e_means, e_vars, e_lambdas, e_times, lambda_0,
                        dt, sigma_phase, sigma_tempo=sigma_tempo, overtime=overtime)
    kb = PATIPPET(kp, drift=True)

    kb.prepare()
    before = time.time()
    kb.run()
    print('Took {}s\n'.format(np.round(time.time() - before, 2)))

