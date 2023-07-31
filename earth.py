import firedrake as fd

# # # === --- constants --- === # # #

# length of a single earth day
day = 24.*60*60
Day = fd.Constant(day)

# radius in metres
radius = 6371220.
Radius = fd.Constant(radius)

# rotation rate
omega = 7.292e-5
Omega = fd.Constant(omega)

# gravitational acceleration
gravity = 9.80616
Gravity = fd.Constant(gravity)

# Coriolis force


def coriolis_expression(x, y, z, omega=Omega, radius=Radius):
    """
    Spatially varying coriolis expression on the sphere.

    :arg x: cartesian x location.
    :arg y: cartesian y location.
    :arg z: cartesian z location.
    :arg omega: rotation rate of the sphere. Defaults to earth.
    :arg radius: radius of the sphere. Defaults to earth.
    """
    return 2*omega*z/radius


def cart_to_sphere_coords(x, y, z):
    '''
    Convert cartesian coordinates to latitude/longitude coordinates.
    :arg x: cartesian x location.
    :arg y: cartesian y location.
    :arg z: cartesian z location.
    '''
    r = fd.sqrt(x*x + y*y + z*z)
    zr = z/r
    # avoid roundoff errors at poles
    zr_corr = fd.min_value(fd.max_value(zr, -1), 1)
    theta = fd.asin(zr_corr)
    lamda = fd.atan_2(y, x)
    return theta, lamda


def sphere_to_cart_vector(x, y, z, uzonal, umerid):
    '''
    Convert vector in spherical coordinates to cartesian coordinates.

    :arg x: cartesian x location.
    :arg y: cartesian y location.
    :arg z: cartesian z location.
    :arg uzonal: zonal component of vector.
    :arg umerid: meridional component of vector.
    '''
    theta, lamda = cart_to_sphere_coords(x, y, z)
    cart_u_expr = -uzonal*fd.sin(lamda) - umerid*fd.sin(theta)*fd.cos(lamda)
    cart_v_expr = uzonal*fd.cos(lamda) - umerid*fd.sin(theta)*fd.sin(lamda)
    cart_w_expr = umerid*fd.cos(theta)
    return fd.as_vector((cart_u_expr, cart_v_expr, cart_w_expr))
