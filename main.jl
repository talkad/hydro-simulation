# import Pkg; Pkg.add("Plots")
ENV["GKSwstype"] = "100" 
using Plots
using SparseArrays
using LinearAlgebra
using SuiteSparse
gr()

include("initial_condition.jl")
using .ic

include("cfd.jl")
using .cfd


function apply_BC!(u, v, U_lid)
    u[1, :] .= 0.0
    u[end, :] .= 0.0

    u[:, 1] .= 0.0
    u[:, end] .= U_lid

    v[:, 1] .= 0.0
    v[:, end] .= 0.0

    v[1, :] .= 0.0
    v[end, :] .= 0.0
end


function calculate_stable_dt(u, v, grid, nu; CFL=0.5, safety_diff=0.2)
    dx, dy = grid.dx, grid.dy
    u_max = maximum(abs.(u)) + 1e-10
    v_max = maximum(abs.(v)) + 1e-10
    
    dt_adv = CFL / (u_max/dx + v_max/dy)
    # The diffusion term is implicitly handled, so only the convection term is considered.
    # dt_diff = safety_diff * (min(dx, dy)^2) / nu
    
    # return min(dt_adv, dt_diff)

    return dt_adv
end


function main()
    Nx, Ny = 64, 64 
    rho = 1
    nu = 0.01
    U_lid = 5.0
    total_time = 5.0
    
    grid, u, v, p, u_star, v_star = initial_condition(Nx, Ny, U_lid=U_lid)
    dx, dy = grid.dx, grid.dy

    t = 0.0
    step = 0
    while t < total_time
        dt = calculate_stable_dt(u, v, grid, nu)

        # Step A: Get Intermediate Velocity (Predictor)
        u_star, v_star = intermediate_velocity(u, v, Nx, Ny, dx, dy, dt, nu)
        apply_BC!(u_star, v_star, U_lid)

        # Step B: Solve Pressure Poisson Equation
        p = solve_pressure(u_star, v_star, Nx, Ny, dx, dy, rho, dt)

        # Step C: Update Velocity (Corrector)
        u, v = update_velocity(u_star, v_star, p, Nx, Ny, dx, dy, rho, dt)
        apply_BC!(u, v, U_lid)

        t += dt
        step += 1
        
        if step % 5 == 0
            u_c = 0.5 .* (u[1:Nx, :] .+ u[2:Nx+1, :])
            v_c = 0.5 .* (v[:, 1:Ny] .+ v[:, 2:Ny+1])
            
            vel_mag = sqrt.(u_c.^2 .+ v_c.^2)

            p_heat = heatmap(grid.x_p, grid.y_p, vel_mag', 
                            c = :viridis, 
                            title = "Velocity Magnitude (Step $step, t=$(round(t, digits=3)))",
                            aspect_ratio = 1,
                            xlims = (0, 1), ylims = (0, 1))
            
            filename = "frames/frame_$(lpad(step, 5, "0")).png"
            savefig(p_heat, filename)
        end
    end

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end