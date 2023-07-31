import firedrake as fd
from collections import namedtuple

GeoParams = namedtuple("GeoParams", "g b f")
MoistPhysics = namedtuple("MoistPhysics", "source beta1 beta2")


def function_space(mesh, degree=1, moisture=False, bouyancy=False):
    """
    Generate the (BDM * DG) compatible finite element space for the
    shallow water equations.

    :arg mesh: the domain.
    :arg degree: the degree of the DG depth space.
    :arg moisture: whether moisture is included in the equation set.
    :arg bouyancy: whether bouyancy is included in the equation set.
    """
    Vu = fd.FunctionSpace(mesh, "BDM", degree+1)
    Vh = fd.FunctionSpace(mesh, "DG", degree)
    W = Vu*Vh

    if bouyancy:
        Vb = Vh
        W *= Vb

    if moisture:
        Vq = Vh
        W *= Vq

    return W


# mass matrices


def mass_matrix(u, v, mesh):
    """
    The standard inner product mass matrix.

    :arg u: the (trial) function.
    :arg v: the test function.
    :arg mesh: the domain.
    """
    return fd.inner(u, v)*fd.dx


form_mass_u = mass_matrix
form_mass_h = mass_matrix
form_mass_q = mass_matrix
form_mass_b = mass_matrix


def form_mass(u, h, q, v, phi, psi, mesh, moisture=False):
    """
    Form the mass matrices for the shallow water equations.

    :arg u: velocity function.
    :arg h: depth function.
    :arg q: moisture function.
    :arg v: velocity test function.
    :arg phi: depth test function.
    :arg psi: moisture test function.
    """
    mass = \
        form_mass_u(u, v, mesh) + \
        form_mass_h(h, phi, mesh)
    if moisture:
        mass += form_mass_q(q, psi, mesh)
    return mass


# DG advection forms


def conservative_advection_form(u, q, test, mesh):
    """
    DG form for the conservative advection of a scalar q:
    div(u*q)

    :arg u: the advecting velocity.
    :arg q: the advected scalar.
    :arg test: the test function for the scalar.
    :arg mesh: the domain.
    """
    n = fd.FacetNormal(mesh)
    uup = 0.5 * (fd.dot(u, n) + abs(fd.dot(u, n)))

    volume = -fd.inner(fd.grad(test), u)*q*fd.dx
    surface = fd.jump(test)*(uup('+')*q('+') - uup('-')*q('-'))*fd.dS
    return volume + surface


# velocity equation forms


def dry_function_u(u, h, v, mesh, gparam, perp=fd.cross):
    """
    Form for the velocity equation of the dry shallow water equations.

    :arg u: velocity function.
    :arg h: depth function.
    :arg q: moisture function.
    :arg v: velocity test function.
    :arg mesh: the domain.
    :arg gparam: a GeoParams object.
    :arg perp: the perp function to use.
        Defaults to perp (correct choice for the sphere).
    """
    n = fd.FacetNormal(mesh)
    outward_normals = fd.CellNormal(mesh)

    g = gparam.g
    b = gparam.b
    f = gparam.f

    def prp(u):
        return perp(outward_normals, u)

    def both(u):
        return 2*fd.avg(u)

    K = 0.5*fd.inner(u, u)
    upwind = 0.5 * (fd.sign(fd.dot(u, n)) + 1)

    return (fd.inner(v, f*prp(u))*fd.dx
            - fd.inner(prp(fd.grad(fd.inner(v, prp(u)))), u)*fd.dx
            + fd.inner(both(prp(n)*fd.inner(v, prp(u))),
                       both(upwind*u))*fd.dS
            - fd.div(v)*(g*(h + b) + K)*fd.dx)


def moist_function_u(u, h, q, v, mesh, gparam, mphys, perp=fd.cross):
    """
    Form for the velocity equation of the moist shallow water equations.

    :arg u: velocity function.
    :arg h: depth function.
    :arg q: moisture function.
    :arg v: velocity test function.
    :arg mesh: the domain.
    :arg gparam: a GeoParams object.
    :arg mphys: a MoistPhysics object.
    :arg perp: the perp function to use.
        Defaults to perp (correct choice for the sphere).
    """
    return dry_function_u(u, h, v, mesh, gparam, perp=perp)


def form_function_u(u, h, q, v, mesh, gparam, mphys,
                    perp=fd.cross, moisture=False):
    """
    The finite-element form for the velocity equation of
    the shallow water equations.

    :arg u: velocity function.
    :arg h: depth function.
    :arg q: moisture function.
    :arg v: velocity test function.
    :arg mesh: the domain.
    :arg gparam: a GeoParams object.
    :arg mphys: a MoistPhysics object.
    :arg perp: the perp function to use.
        Defaults to perp (correct choice for the sphere).
    :arg moisture: whether moisture is included in the equations.
    """
    if moisture:
        function = moist_function_u(u, h, q, v, mesh, gparam, mphys, perp=perp)
    else:
        function = dry_function_u(u, h, v, mesh, gparam, perp=perp)
    return function


# depth equation forms


def dry_function_h(u, h, phi, mesh, gparam):
    """
    Form for the depth equation of the dry shallow water equations.

    :arg u: velocity function.
    :arg h: depth function.
    :arg q: moisture function.
    :arg phi: depth test function.
    :arg mesh: the domain.
    :arg gparam: a GeoParams object.
    """
    return conservative_advection_form(u, h, phi, mesh)


def moist_function_h(u, h, q, phi, mesh, gparam, mphys):
    """
    Form for the depth equation of the moist shallow water equations.

    :arg u: velocity function.
    :arg h: depth function.
    :arg q: moisture function.
    :arg phi: depth test function.
    :arg mesh: the domain.
    :arg gparam: a GeoParams object.
    :arg mphys: a MoistPhysics object.
    """
    dry_function = dry_function_h(u, h, phi, mesh, gparam)

    beta1 = mphys.beta1
    p = mphys.source(q, phi)
    source = -beta1*p*fd.dx

    return dry_function - source


def form_function_h(u, h, q, phi, mesh, gparam, mphys, moisture=False):
    """
    The finite-element form for the depth equation of
    the shallow water equations.

    :arg u: velocity function.
    :arg h: depth function.
    :arg q: moisture function.
    :arg phi: depth test function.
    :arg mesh: the domain.
    :arg gparam: a GeoParams object.
    :arg mphys: a MoistPhysics object.
    :arg moisture: whether moisture is included in the equations.
    """
    if moisture:
        function = moist_function_h(u, h, q, phi, mesh, gparam, mphys)
    else:
        function = dry_function_h(u, h, q, phi, mesh, gparam)
    return function


# moisture equation forms


def form_function_q(u, h, q, psi, mesh, gparam, mphys):
    """
    The finite-element form for the moisture equation of the
    shallow water equations.

    :arg u: velocity function.
    :arg h: depth function.
    :arg q: moisture function.
    :arg psi: moisture test function.
    :arg mesh: the domain.
    :arg gparam: a GeoParams object.
    :arg mphys: a MoistPhysics object.
    """
    advection = conservative_advection_form(u, q, psi, mesh)

    p = mphys.source(q, psi)
    source = -p*fd.dx

    return advection - source


# total function


def form_function(u, h, q, v, phi, psi, mesh, gparam, mphys, moisture=False):
    """
    The finite-element form for the shallow water equations.

    :arg u: velocity function.
    :arg h: depth function.
    :arg q: moisture function.
    :arg v: velocity test function.
    :arg phi: depth test function.
    :arg psi: moisture test function.
    :arg mesh: the domain.
    :arg gparam: a GeoParams object.
    :arg mphys: a MoistPhysics object.
    :arg moisture: whether moisture is included in the equations.
    """
    function = \
        form_function_u(u, h, q, v, mesh, gparam, mphys, moisture=moisture) + \
        form_function_h(u, h, q, phi, mesh, gparam, mphys, moisture=moisture)
    if moisture:
        function += form_function_q(u, h, q, psi, mesh, gparam, mphys)
    return function
