
from collections import namedtuple

RungeKuttaCoeffs = namedtuple("RungeKuttaCoeffs", "a b d cfl")


def rungekutta_coeffs(method='34'):
    """
    The coefficients of an mth-stage Runge-Kutta scheme of the form:
    u_{l} = a_{k}*u_{n} + b_{k}*w_{k-1}
    u_{k} = u_{l} + d*dt*L(u_{l})
    u_{n+1} = u_{m}

    :arg method: string of two numbers:
        method[0]: order of accuracy
        method[1]: number of stages
    """
    # third order three stage
    if method == '33':
        d = 1.
        a = [0, 3./4, 1./3]
        b = [1, 1./4, 2./3]
        cfl = 1.

    # third order four stage
    elif method == '34':
        d = 0.5
        a = [0, 0, 2./3, 0]
        b = [1, 1, 1./3, 1]
        cfl = 2.

    return RungeKuttaCoeffs(a, b, d, cfl)


class SSPRK(object):
    """
    A strong stability preserving Runge-Kutta scheme.
    Given a set of Runge-Kutta coefficients and a solver to
    take one forward euler step, will take a single timestep
    of the Runge-Kutta scheme.
    """
    def __init__(self, wl, wr, solver, coeffs,
                 pre_solve_callback=lambda wr: None,
                 post_solve_callback=lambda wl: None):
        """
        :arg wl: the Function on the left-hand-side of the Euler forward
            solver (the unknown solution at the next timestep/stage).
        :arg wr: the Function on the right-hand-side of the Euler forward
            solver (the known solution at the current timestep/stage).
        :arg solver: a solver to take a single Euler forward step.
        :arg coeffs: a RungeKuttaCoeffs instance with the coefficients
            for an SSPRK scheme.
        :arg pre_solve_callback: a callable taking the solution at the
            beginning of the stage as an argument. This callback is evaluated
            before each application of the Euler forward solver.
        :arg post_solve_callback: a callable taking the calculated solution
            at the end of the stage as an argument. This callback is evaluated
            after each application of the Euler forward solver.
        """

        self.wl = wl
        self.wr = wr
        self.solver = solver

        self.coeffs = coeffs

        self.pre_solve_callback = pre_solve_callback
        self.post_solve_callback = post_solve_callback

    def __call__(self, wn, wn1):
        """
        Take one timestep of the SSPRK scheme.

        :arg wn: the solution at the beginning of the timestep.
        :arg wn1: the calculated solution at the end of the timestep.
        """
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
