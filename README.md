# Lid-Driven Cavity Solver

A finite-difference implementation of the two-dimensional incompressible lid-driven cavity problem.

## Governing Equations

The solver solves the incompressible Navier–Stokes equations:

**Momentum**

```math
\frac{\partial \mathbf{u}}{\partial t}
+ (\mathbf{u}\cdot\nabla)\mathbf{u}
=
-\frac{1}{\rho}\nabla p
+\nu\nabla^2\mathbf{u}
```

**Continuity**

```math
\nabla\cdot\mathbf{u}=0
```

where $\mathbf{u}$ is the velocity, $p$ is the pressure, $\rho$ is the density, and $\nu$ is the kinematic viscosity.

## Solver Flow

```text
Initialize fields
        │
        ▼
Choose time discretization
(Backward Euler / BDF2)
        │
        ▼
Solve momentum equations
        │
        ▼
Pressure correction
(SIMPLE / PISO)
        │
        ▼
Enforce incompressibility
        │
        ▼
Advance to next time step
```

## Visualization

<p align="center"> <img src="frames/output.gif" width="600" alt="Lid-driven cavity simulation"> </p>
