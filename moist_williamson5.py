import firedrake as fd

import williamson5 as wcase
import earth
import shallow_water as swe
from rungekutta import rungekutta_coeffs, SSPRK
from physics import MoistPhysics, SaturationSource

from functools import partial

from firedrake.petsc import PETSc
Print = PETSc.Sys.Print

moisture = True

# set up the domain

mesh = fd.IcosahedralSphereMesh(radius=earth.radius,
                                refinement_level=3)
x = fd.SpatialCoordinate(mesh)
mesh.init_cell_orientations(x)

# set up the functionspaces

W = swe.function_space(mesh, moisture=moisture)
Vu, Vh, Vq = W.subfunctions

# physical parameters

gparam = swe.GeoParams(
    earth.Gravity,
    fd.Function(Vh).interpolate(wcase.topography_expression(*x)),
    fd.Function(Vh).interpolate(wcase.coriolis_expression(*x))
)

dt = 600.
dT = fd.Constant(dt)

# convert between total depth and perturbation


def elevation(eta, h):
    return eta.interpolate(h + gparam.b - wcase.H0)


def depth(h, eta):
    return h.interpolate(wcase.H0 + eta - gparam.b)


# the unknowns and the forms

# moisture parameters
# saturation_curve = fd.Constant(10e-4)
gamma = fd.Constant(10e-3)
tau = dT
qref = fd.Constant(3.)
alpha = fd.Constant(-0.6)

# left/right hand side functions
wl = fd.Function(W)
wr = fd.Function(W)

# initial conditions
winitial = fd.Function(W)

uinit, hinit, qinit = winitial.subfunctions
uinit.interpolate(wcase.velocity_expression(*x))
depth(hinit, wcase.elevation_expression(*x))
qinit.assign(0.9*qref)

w0 = winitial.copy(deepcopy=True)
w1 = winitial.copy(deepcopy=True)


# the moisture source term
def saturation_curve(win):
    H = wcase.H0
    h = win.subfunctions[1]
    return qref*fd.exp(-alpha*(h-H)/H)


saturation_source = SaturationSource(Vq, saturation_curve,
                                     gamma=gamma, tau=tau,
                                     constant_saturation_curve=False,
                                     method='source')

mphys = MoistPhysics(source=saturation_source.source_term,
                     beta1=fd.Constant(1),
                     beta2=fd.Constant(0))

pre_solve_callback = saturation_source.update

# form generating functions

form_mass = partial(swe.form_mass, mesh=mesh, moisture=moisture)
form_function = partial(swe.form_function, mesh=mesh, gparam=gparam,
                        mphys=mphys, moisture=moisture)

# RK weighting on timestep
method = '34'
rkcoeffs = rungekutta_coeffs(method=method)
dcoeff = rkcoeffs.d

# function only on the rhs for explicit RK schemes

v = fd.TestFunctions(W)

lhs = form_mass(*fd.split(wl), *v)
rhs = form_mass(*fd.split(wr), *v) - dcoeff*dT*form_function(*fd.split(wr), *v)
form = lhs - rhs

sparams = {
    'snes_type': 'ksponly',
    'snes': {
        'monitor': None,
        'converged_reason': None,
    },
    'ksp': {
        'monitor': None,
        'converged_reason': None,
    },
    'ksp_type': 'preonly',
    'pc_type': 'lu'
}

problem = fd.NonlinearVariationalProblem(form, wl)
solver = fd.NonlinearVariationalSolver(problem, solver_parameters=sparams)

# Runge-Kutta stepper
ssprk = SSPRK(wl, wr, solver, rkcoeffs,
              pre_solve_callback=pre_solve_callback)

# write some output
wout = winitial.copy(deepcopy=True)
uout, hout, qout = wout.subfunctions
eta = fd.Function(Vh, name="elevation")
uout.rename("velocity")
hout.rename("elevation")
qout.rename("moisture")
ofile = fd.File("results/moist_williamson5/williamson5.pvd")


def write(wo, t):
    wout.assign(wo)
    ofile.write(uout, elevation(eta, hout), qout, t=t)


write(w0, t=0.)

# lets go

nsteps = 144
for step in range(nsteps):
    Print(f"\n--- Timestep {step} ---\n")
    ssprk(w0, w1)
    w0.assign(w1)
    write(w1, t=dt*(step+1.)/(60*60))
