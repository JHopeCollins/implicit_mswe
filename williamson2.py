from math import pi

import firedrake as fd
import earth

# # # === --- constants --- === # # #

# gravitational constant * reference depth
gh0 = 2.94e4
Gh0 = fd.Constant(gh0)

# reference depth
h0 = gh0/earth.gravity
H0 = fd.Constant(h0)

# days taken for velocity to travel circumference
period = 12.0
Period = fd.Constant(period)

# reference velocity
u0 = 2*pi*earth.radius/(period*earth.day)
U0 = fd.Constant(u0)

# # # === --- analytical solution --- === # # #


# coriolis parameter f
def coriolis_expression(x, y, z):
    """
    UFL expression for the coriolis.

    :arg x: cartesian x location.
    :arg y: cartesian y location.
    :arg z: cartesian z location.
    """
    return earth.coriolis_expression(x, y, z)


def velocity_expression(x, y, z, uref=U0):
    """
    UFL expression for the initial velocity.

    :arg x: cartesian x location.
    :arg y: cartesian y location.
    :arg z: cartesian z location.
    :arg uref: the reference velocity. Defaults to the Williamson1992 value.
    """
    return fd.as_vector([-uref*y/earth.Radius, uref*x/earth.Radius, 0.0])


# elevation field eta
def elevation_expression(x, y, z, href=H0, uref=U0):
    """
    UFL expression for the initial elevation perturbation.

    :arg x: cartesian x location.
    :arg y: cartesian y location.
    :arg z: cartesian z location.
    :arg uref: the reference velocity. Defaults to the Williamson1992 value.
    :arg href: the reference depth. Defaults to the Williamson1992 value.
    """
    z0 = z/earth.Radius
    k = (earth.Radius*earth.Omega*uref + uref*uref/2.0)
    return - k*(z0*z0)/earth.Gravity


# topography field b
def topography_expression(x, y, z):
    """
    UFL expression for the topography.

    :arg x: cartesian x location.
    :arg y: cartesian y location.
    :arg z: cartesian z location.
    """
    return fd.Constant(0)
