# import Pkg; Pkg.add("FileIO")
# import Pkg; Pkg.add("CairoMakie")
# import Pkg; Pkg.add("Interpolations")

ENV["GKSwstype"] = "100"
using SparseArrays
using LinearAlgebra
using SuiteSparse

include("utils.jl");                      using .utils
include("initial_condition.jl");            using .ic
include("boundary_condition/bc.jl");        using .bc
include("time_discretization/bdf2.jl");          using .bdf2
include("time_discretization/first_order.jl");   using .first_order
include("pressure_solver/simple.jl");         using .simple
include("plot.jl");                         using .plot_utils


function calculate_stable_dt(u, v, grid, nu; CFL=0.5)
    dx, dy = grid.dx, grid.dy
    u_max = maximum(abs.(u)) + 1e-10
    v_max = maximum(abs.(v)) + 1e-10

    # The diffusion term is implicitly handled, so only the convection term is considered.
    # dt_diff = safety_diff * (min(dx, dy)^2) / nu
    # return min(dt_adv, dt_diff)

    return CFL / (u_max/dx + v_max/dy)
end


function main()
    # Set simulation parameters
    time_discretization = "bdf2"

    Nx, Ny  = 124, 124
    rho     = 2.0
    nu      = 0.05
    L = 1.0
    U_lid   = 20.0
    total_time = 5.0
    plot_every = 5

    println("Reynolds number: ", U_lid * L / nu)
    
    grid, u, v, p = initial_condition(Nx, Ny; domain_size=(L, L), U_lid=U_lid)
    dt = calculate_stable_dt(u, v, grid, nu)

    solver = simple.solver

    # Define common operators
    n_u = Ny * (Nx + 1)
    n_v = (Ny + 1) * Nx
    ops = utils.Operators(Nx, Ny, grid.dx, grid.dy)

    A_u_1 = sparse(I, n_u, n_u) - dt * nu * ops.L_u
    A_v_1 = sparse(I, n_v, n_v) - dt * nu * ops.L_v
    Fu_1 = lu(A_u_1)
    Fv_1 = lu(A_v_1)

    A_u_2 = sparse(I, n_u, n_u) - 2dt / 3 * nu * ops.L_u
    A_v_2 = sparse(I, n_v, n_v) - 2dt / 3 * nu * ops.L_v
    Fu_2 = lu(A_u_2)
    Fv_2 = lu(A_v_2)

    # Time-stepping loop
    t    = 0.0
    step = 0

    u_prev = copy(u)
    v_prev = copy(v)

    while t < total_time

        if time_discretization == "first_order" || step == 0
            Fu = Fu_1; Fv = Fv_1
            u_star, v_star = first_order.intermediate_velocity(u, v, p, ops, Nx, Ny, dt, rho, Fu_1, Fv_1)
        else
            Fu = Fu_2; Fv = Fv_2
            u_star, v_star = bdf2.intermediate_velocity(u, u_prev, v, v_prev, p, ops, Nx, Ny, dt, rho, Fu_2, Fv_2)
        end
        apply_vel_derichlet!(u_star, v_star, U_lid)

        u_prev .= u
        v_prev .= v

        u, v, p = time_discretization == "first_order" ? solver(u_star, v_star, p, ops, grid, rho, dt, Fu, Fv) : 
                                                        solver(u_star, v_star, p, ops, grid, rho, dt, Fu, Fv; u_prev=u_prev, v_prev=v_prev)
        apply_vel_derichlet!(u, v, U_lid)
        
        # Debug
        # print(maximum(u), " ", maximum(v), " ", maximum(p), "\n")

        if step % plot_every == 0
            # plot_velocity_magnitude(u, v, grid, step, t)
            plot_streamlines(u, v, grid, step, t)
        end

        t    += dt
        step += 1
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end