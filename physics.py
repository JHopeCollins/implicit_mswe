import firedrake as fd
from collections import namedtuple

MoistPhysics = namedtuple("MoistPhysics", "source beta1 beta2")


class SaturationSource(object):
    """
    Moisture scheme that converts moisture directly into rain with
    feedback onto the depth equation.
    """
    def __init__(self, Vq, saturation_curve, tau,
                 gamma=1., method='source',
                 constant_saturation_curve=True):
        """
        :arg Vq: The moisture function space
        :arg saturation_curve: The moisture level above which moisture
            is converted to rain and the source term is active.
        :arg gamma: The proportion of moisture above the saturation curve
            that is converted to rain.
        :arg tau: The timescale over which the moisture is converted to rain.
        :arg method: Either 'source' or 'reaction'.
            If 'reaction', the source term is built with the moisture argument
                given to SaturationSource.source. This moisture argument should
                be the moisture component from the subject Function. The source
                term will then be included in the equation Jacobian.
            If 'source', the source term is built from a seperate Function held
                by SaturationSource and is updated as necessary. In this case
                the source term will not be included in the equation Jacobian.
        """
        self.function_space = Vq
        self.gamma = gamma
        self.tau = tau
        self.method = method

        self.constant_saturation_curve = constant_saturation_curve
        if constant_saturation_curve:
            self.saturation_curve = saturation_curve
        else:
            self.saturation_curve_calculation = saturation_curve
            self.saturation_curve = fd.Function(Vq)

        if method == 'source':
            self.moisture = fd.Function(self.function_space)
            self.source_function = fd.Function(self.function_space)
            self.interpolator = fd.Interpolator(
                self.source_expr(self.moisture),
                self.function_space)

    def source_expr(self, moisture):
        """
        The UFL expression for the source term.

        :arg moisture: the moisture Function to calculate the source term from.
        """
        expr = fd.conditional(
            moisture > self.saturation_curve,
            (self.gamma/self.tau)*(moisture - self.saturation_curve), 0)
        return expr

    def source_term(self, q):
        """
        Return a value to be used as the source term when building the
        finite-element forms.
        """
        if self.method == 'source':
            source = self.source_function
        else:  # method == 'reaction'
            source = self.source_expr(q)
        return source

    def update(self, w):
        """
        Update the source term.
        If method='source' then the source term will be updated from the given
        Function. If method='reaction' then the source term is assumed to
        always be up to date with the current solution.

        :arg w: the current solution.
        """
        if not self.constant_saturation_curve:
            new_curve = self.saturation_curve_calculation(w)
            self.saturation_curve.interpolate(new_curve)

        if self.method == 'source':
            q = w.subfunctions[2]
            self.moisture.assign(q)
            self.source_function.assign(self.interpolator.interpolate())
