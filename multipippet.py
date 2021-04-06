'''
TODO

----- Contact

Tom Kaplan (Music Cognition Lab, QMUL, UK)
t.m.kaplan@qmul.ac.uk

'''
import attr
import numpy as np

from pippet import PIPPET, PIPPETParams

class PIPPETMulti(object):
    ''' PIPPET ensemble with template plausibility hypotheses '''

    def __init__(self, params, prior=None, drift=False, ensemble=False):
        # Whether to phase-track in each sub-model, or use top-level stats
        self._ensemble = ensemble
        # Create our ensemble
        self.models = []
        for p in params:
            self.models.append(PIPPET(p, drift=drift))
        self.n_m = len(self.models)

        # TODO: Better constraints (or, abstraction) for params. of the sub-models
        assert len(set(m.n_ts for m in self.models)) is 1, 'Different timelines?'
        assert all(not m.p.eta_e for m in self.models), 'Evt perc. noise unsupported!'

        self.n_ts = self.models[0].n_ts
        self.dt = self.models[0].p.dt
        self.ts = self.models[0].ts.copy()

        # Probability distribution
        self.p_m = np.zeros((self.n_ts, self.n_m))
        if prior is not None:
            self.p_m[0] = prior
        else:
            self.p_m[0] = 1
        self.p_m[0] = self.p_m[0]/self.p_m[0].sum()

        # Aggregate sufficient statistics and lambda
        self.phibar_s = np.zeros(self.n_ts)
        self.phibar_s[0] = self.models[0].p.mean0
        self.V_s = np.zeros(self.n_ts)
        self.V_s[0] = self.models[0].p.var0
        self.L_ms = np.zeros((self.n_ts, self.n_m))
        self.L_s = np.zeros(self.n_ts)

    def run(self):
        ''' Run PIPPETMulti models in parallel, updating template plausibility '''

        # Initial Lambda values
        for j, m in enumerate(self.models):
            phibar_m, V_m = m.phibar_s[0], m.V_s[0]
            lambda_m = m.lambda_hat(phibar_m, V_m)
            self.L_ms[0, j] += lambda_m
        self.L_s[0] = np.sum(self.p_m[0] * self.L_ms[0])

        for i in range(1, self.n_ts):
            lambda_prev = self.L_s[i-1]
            phibar_prev = self.phibar_s[i-1]
            V_prev = self.V_s[i-1]

            phibar_ms = np.zeros(self.n_m)
            V_ms = np.zeros(self.n_m)

            # Single time step for each model in ensemble
            for j, m in enumerate(self.models):
                if self._ensemble:
                    phibar_m, V_m = m.step(i)
                else:
                    phibar_m, V_m = m.step(i, phibar_prev, V_prev)
                lambda_m = m.lambda_hat(phibar_m, V_m)

                # Update p_m for each template/model
                prev_p_m = self.p_m[i-1, j]
                d_p_m = prev_p_m * (lambda_m/lambda_prev - 1) * lambda_m
                self.p_m[i, j] = prev_p_m + self.dt * (1 - d_p_m)

                self.L_ms[i, j] = lambda_m
                phibar_ms[j] = phibar_m
                V_ms[j] = V_m

            # Normalise our template plausibility distribution
            self.p_m[i] /= self.p_m[i].sum()

            self.phibar_s[i] = np.sum(self.p_m[i] * phibar_ms)
            self.V_s[i] = np.sum(self.p_m[i] * V_ms)
            self.L_s[i] = np.sum(self.p_m[i] * self.L_ms[i])

if __name__ == "__main__":
    import time
    #Example: PIPPETMulti

    # Rhythm around 2:2:3
    e_times = np.array([0.0, 0.189, 0.392, 0.7])
    lambda_0 = 0.01
    dt = 0.001
    sigma_phase = 0.05
    overtime = 0.1

    m1 = '1:1:1'
    ioi = 0.7/3
    e_means1 = np.arange(0, len(e_times)) * ioi
    e_vars1 = np.array([0.0001]).repeat(len(e_means1))
    e_lambdas1 = np.array([0.05]).repeat(len(e_means1))
    kp1 = PIPPETParams(e_means1, e_vars1, e_lambdas1,
                       e_times, lambda_0, dt, sigma_phase, overtime=overtime, t0=0)
    m2 = '2:2:3'
    ioi = 0.7/7
    e_means2 = np.array([0.0, ioi*2, ioi*4, ioi*7])
    e_vars2 = np.array([0.0001]).repeat(len(e_means2))
    e_lambdas2 = np.array([0.05]).repeat(len(e_means2))
    kp2 = PIPPETParams(e_means2, e_vars2, e_lambdas2,
                       e_times, lambda_0, dt, sigma_phase, overtime=overtime, t0=0)
    m3 = '1:1:2'
    ioi = 0.7/4
    e_means3 = np.array([0.0, ioi, ioi*2, ioi*4])
    e_vars3 = np.array([0.0001]).repeat(len(e_means3))
    e_lambdas3 = np.array([0.05]).repeat(len(e_means3))
    kp3 = PIPPETParams(e_means3, e_vars3, e_lambdas3,
                       e_times, lambda_0, dt, sigma_phase, overtime=overtime, t0=0)
    prior = np.array([1.0, 1.0, 1.0])/3.
    modelnames = [m1, m2, m3]
    ms = [kp1, kp2, kp3]

    model = PIPPETMulti(ms, prior=prior, drift=True, ensemble=False)
    before = time.time()
    model.run()
    print('Took {}s\n'.format(np.round(time.time() - before, 2)))

    import matplotlib.pyplot as plt
    from pippet_plots import plot_multipippet_all, plot_multipippet_internals
    plot_multipippet_all(model, modelnames)
    plot_multipippet_internals(model, modelnames)
    plt.show()

