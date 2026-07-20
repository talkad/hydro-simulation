# The immersed boundary method with SIMPLE

A finite-difference implementation of two-dimensional incompressible channel flow past a cylinder, using an immersed boundary method.

## Governing Equations

The solver solves the incompressible Navier–Stokes equations:

**Momentum**

```math
\frac{\partial \mathbf{u}}{\partial t}
+ (\mathbf{u}\cdot\nabla)\mathbf{u}
=
-\frac{1}{\rho}\nabla p
+\nu\nabla^2\mathbf{u} + \mathbf{F}
```

**Continuity**

```math
\nabla\cdot\mathbf{u}=0
```

**No-Slip**

```math
u_s = U_s
```

where $\mathbf{u}$ is the velocity, $p$ is the pressure, $\mathbf{F}$ is the immersed boundary force, $\rho$ is the density, $\nu$ is the kinematic viscosity, and $U_s$ is the prescribed surface velocity.

## Solver Flow

```text
Initialize fields
        │
        ▼
Choose time discretization
       BDF2
        │
        ▼
Solve momentum equations 
        │
        ▼
Pressure and Forces correction
     SIMPLE
        │
        ▼
Enforce incompressibility
        │
        ▼
Advance to next time step
```

## Visualization

<p align="center"> <img src="gifs/u.gif" width="1000" alt="Horizontal velocity field"> </p>

<p align="center"> <img src="gifs/v.gif" width="1000" alt="Vertical velocity field"> </p>
