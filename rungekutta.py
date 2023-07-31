
from collections import namedtuple

RungeKuttaCoeffs = namedtuple("RungeKuttaCoeffs", "a b d")


def rungekutta_coeffs(method='34'):
    # third order three stage
    if method == '33':
        d = 1
        a = [0, 3./4, 1./3]
        b = [1, 1./4, 2./3]

    # third order four stage
    elif method == '34':
        d = 0.5
        a = [0, 0, 2./3, 0]
        b = [1, 1, 1./3, 1]

    return RungeKuttaCoeffs(a, b, d)


class SSPRK(object):
    def __init__(self, wl, wr, solver, coeffs,
                 pre_solve_callback=lambda wr: None,
                 post_solve_callback=lambda wl: None):

        self.wl = wl
        self.wr = wr
        self.solver = solver

        self.coeffs = coeffs

        self.pre_solve_callback = pre_solve_callback
        self.post_solve_callback = post_solve_callback

    def __call__(self, wn, wn1):
        wl = self.wl
        wr = self.wr

        wr.assign(wn)
        for a, b in zip(self.coeffs.a, self.coeffs.b):

            self.pre_solve_callback(wr)

            wl.assign(wr)
            self.solver.solve()

            self.post_solve_callback(wl)

            wr.assign(a*wn + b*wl)

        wn1.assign(wr)
        return wn1
