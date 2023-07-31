import firedrake as fd
from collections import namedtuple

GeoParams = namedtuple("GeoParams", "g b f")
MoistPhysics = namedtuple("MoistPhysics", "source beta1 beta2")


def function_space(mesh, degree=1, moisture=False, bouyancy=False):
    Vu = fd.FunctionSpace(mesh, "BDM", degree+1)
    Vh = fd.FunctionSpace(mesh, "DG", degree)
    W = Vu*Vh

    if moisture:
        Vq = Vh
        W *= Vq

    if bouyancy:
        Vb = Vh
        W *= Vb

    return W


# mass matrices


def mass_matrix(u, v, mesh):
    return fd.inner(u, v)*fd.dx


form_mass_u = mass_matrix
form_mass_h = mass_matrix
form_mass_q = mass_matrix
form_mass_b = mass_matrix


def form_mass(u, h, q, v, phi, psi, mesh):
    mass = \
        form_mass_u(u, v, mesh) + \
        form_mass_h(h, phi, mesh) + \
        form_mass_q(q, psi, mesh)
    return mass


# DG conservative advection form


def conservative_advection_form(u, q, test, mesh):
    n = fd.FacetNormal(mesh)
    uup = 0.5 * (fd.dot(u, n) + abs(fd.dot(u, n)))

    volume = -fd.inner(fd.grad(test), u)*q*fd.dx
    surface = fd.jump(test)*(uup('+')*q('+') - uup('-')*q('-'))*fd.dS
    return volume + surface


# velocity equation forms


def dry_function_u(u, h, v, mesh, gparam, perp=fd.cross):
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
    return dry_function_u(u, h, v, mesh, gparam, perp=perp)


def form_function_u(u, h, q, v, mesh, gparam, mphys,
                    perp=fd.cross, moisture=False):
    if moisture:
        function = moist_function_u(u, h, q, v, mesh, gparam, mphys, perp=perp)
    else:
        function = dry_function_u(u, h, v, mesh, gparam, perp=perp)
    return function


# depth equation forms


def dry_function_h(u, h, phi, mesh, gparam):
    return conservative_advection_form(u, h, phi, mesh)


def moist_function_h(u, h, q, phi, mesh, gparam, mphys):
    dry_function = dry_function_h(u, h, phi, mesh, gparam)

    beta1 = mphys.beta1
    p = mphys.source(q, phi)
    source = -beta1*p*fd.dx

    return dry_function - source


def form_function_h(u, h, q, phi, mesh, gparam, mphys, moisture=False):
    if moisture:
        function = moist_function_h(u, h, q, phi, mesh, gparam, mphys)
    else:
        function = dry_function_h(u, h, q, phi, mesh, gparam)
    return function


# moisture equation forms


def form_function_q(u, h, q, psi, mesh, gparam, mphys):
    advection = conservative_advection_form(u, q, psi, mesh)

    p = mphys.source(q, psi)
    source = -p*fd.dx

    return advection - source


def form_function(u, h, q, v, phi, psi, mesh, gparam, mphys, moisture=False):
    function = \
        form_function_u(u, h, q, v, mesh, gparam, mphys, moisture=moisture) + \
        form_function_h(u, h, q, phi, mesh, gparam, mphys, moisture=moisture)
    if moisture:
        function += form_function_q(u, h, q, psi, mesh, gparam, mphys)
    return function
