# import Pkg; Pkg.add("FileIO")
# import Pkg; Pkg.add("CairoMakie")
# import Pkg; Pkg.add("Interpolations")
# import Pkg; Pkg.add("IterativeSolvers")

ENV["GKSwstype"] = "100"
using SparseArrays
using LinearAlgebra
using SuiteSparse

include("utils.jl");                             using .utils
include("initial_condition.jl");                 using .ic
include("boundary_condition/bc.jl");             using .bc
include("time_discretization/bdf2.jl");          using .bdf2
include("time_discretization/first_order.jl");   using .first_order
include("pressure_solver/simple.jl");            using .simple
include("plot.jl");                              using .plot_utils


function calculate_stable_dt(u, v, grid; CFL=0.5)
    # The convective term handles explicit while the diffusive term is handled implicitly. 
    # Therefore, the CFL condition is based on the convective term only.
    h = grid.h
    u_max = maximum(abs.(u)) + 1e-10
    v_max = maximum(abs.(v)) + 1e-10

    return CFL / (u_max/h + v_max/h)
end


function main()
    # Set simulation parameters
    h = 0.1
    nu      = 0.01
    Lx, Ly = 22, 4.1
    Nx, Ny = round(Int, Lx / h), round(Int, Ly / h)

    U_max   = 5
    total_time = 40.0
    plot_every = 5

    cylinder_obj = (2, 2, 0.5)  # (x_center, y_center, radius)

    println("Reynolds number: ", 2 * cylinder_obj[3] * U_max / nu)
    
    horizontal_velocity = horizontal_velocity_profile(Ny, U_max)
    grid, u, v, p, u_star, v_star, cylinder_lagrangian = initial_condition(Nx, Ny, horizontal_velocity, cylinder_obj; domain_size=(Lx, Ly))
    dt = calculate_stable_dt(u, v, grid; CFL=0.25)

    # Define common operators
    n_u = Ny * (Nx + 1)
    n_v = (Ny + 1) * Nx
    ops = utils.build_operators(Nx, Ny, grid.h)

    A_u = sparse(I, n_u, n_u) - (2dt / 3) * nu * ops.L_u    # Second order backward differentiation formula (BDF2) for time discretization
    A_v = sparse(I, n_v, n_v) - (2dt / 3) * nu * ops.L_v
    A_u = lu(A_u)
    A_v = lu(A_v)

    # Interpolation and spreading operators for the immersed boundary
    Iu, Ru, Iv, Rv = ib_operators(cylinder_lagrangian, grid)
    ib_schur = ib_setup(ops, Iu, Ru, Iv, Rv)

    # Time-stepping loop
    t    = 0.0
    step = 0

    u_prev = copy(u)
    v_prev = copy(v)

    # Lagrangian forces
    F = zeros(length(cylinder_lagrangian.X), 2)  # (Fx, Fy) for each marker

    while t < total_time

        if step == 0
            u_star, v_star = first_order.intermediate_velocity(u, v, p, ops, Nx, Ny, dt, 
                                                                sparse(I, n_u, n_u) - dt * nu * ops.L_u, 
                                                                sparse(I, n_v, n_v) - dt * nu * ops.L_v)  # Lagrangian forces at t=0 are zero
        else
            u_star, v_star = bdf2.intermediate_velocity(u, u_prev, v, v_prev, p, ops, Nx, Ny, dt, A_u, A_v, F, Ru, Rv)
        end
        apply_vel_derichlet!(u_star, v_star, horizontal_velocity)

        u_prev .= u
        v_prev .= v

        # Debug
        print(maximum(u[2:40, 2:40]), "  ", maximum(v[2:40, 2:40]), "  ", maximum(p), "\n")

        u, v, delta_p, delta_Fx, delta_Fy = simple.simple_step(ib_schur, ops, Iu, Ru, Iv, Rv, u_star, v_star, dt; u_body=0, v_body=0, verbose=false)
        apply_vel_derichlet!(u, v, horizontal_velocity)
        p .+= reshape(delta_p, size(p))
        F[:, 1] .+= reshape(delta_Fx, size(F[:, 1]))
        F[:, 2] .+= reshape(delta_Fy, size(F[:, 2]))

        if step % plot_every == 0
            plot_streamlines(u, v, grid, step, t, cylinder_lagrangian)
            plot_horizontal_velocity(u, grid, step, t, cylinder_lagrangian)
            plot_vertical_velocity(v, grid, step, t, cylinder_lagrangian)
            plot_pressure(p, grid, step, t, cylinder_lagrangian)
        end

        t    += dt
        step += 1
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

