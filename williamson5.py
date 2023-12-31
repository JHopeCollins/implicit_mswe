from math import pi

import firedrake as fd
import earth
import williamson2 as case2

# # # === --- constants --- === # # #

# reference depth
h0 = 5960.
H0 = fd.Constant(h0)

# reference velocity
u0 = 20.
U0 = fd.Constant(u0)

# mountain parameters
mountain_height = 2000.
Mountain_height = fd.Constant(mountain_height)
mountain_radius = pi/9.
Mountain_radius = fd.Constant(mountain_radius)

# different lambda_c because atan_2 used for angle
mountain_centre_lambda = -pi/2.
Mountain_centre_lambda = fd.Constant(mountain_centre_lambda)

mountain_centre_theta = pi/6.
Mountain_centre_theta = fd.Constant(mountain_centre_theta)

# # # === --- analytical solution --- === # # #


# coriolis parameter f
def coriolis_expression(x, y, z):
    """
    UFL expression for the coriolis.

    :arg x: cartesian x location.
    :arg y: cartesian y location.
    :arg z: cartesian z location.
    """
    return case2.coriolis_expression(x, y, z)


# velocity field u
def velocity_expression(x, y, z, uref=U0):
    """
    UFL expression for the initial velocity.

    :arg x: cartesian x location.
    :arg y: cartesian y location.
    :arg z: cartesian z location.
    :arg uref: the reference velocity. Defaults to the Williamson1992 value.
    """
    return case2.velocity_expression(x, y, z, uref=uref)


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
    return case2.elevation_expression(x, y, z, href=href, uref=uref)


# topography field b
def topography_expression(x, y, z,
                          radius=Mountain_radius,
                          height=Mountain_height,
                          theta_c=Mountain_centre_theta,
                          lambda_c=Mountain_centre_lambda):
    """
    UFL expression for the topography.

    :arg x: cartesian x location.
    :arg y: cartesian y location.
    :arg z: cartesian z location.
    :arg radius: radius of the base of the mountain.
        Defaults to the Williamson1992 value.
    :arg height: height of the mountain. Defaults to the Williamson1992 value.
    :arg theta_c: latitude of the centre of the mountain.
        Defaults to the Williamson1992 value.
    :arg lambda_c: longitude of the centre of the mountain.
        Defaults to the Williamson1992 value.
    """

    lambda_x = fd.atan_2(y/earth.Radius, x/earth.Radius)
    theta_x = fd.asin(z/earth.Radius)

    radius2 = pow(radius, 2)
    lambda2 = pow(lambda_x - lambda_c, 2)
    theta2 = pow(theta_x - theta_c, 2)

    min_arg = fd.min_value(radius2, theta2 + lambda2)

    return height*(1 - fd.sqrt(min_arg)/radius)
