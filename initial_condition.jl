module ic

    export initial_condition, horizontal_velocity_profile

    struct PhysicalGrid
        Nx::Int
        Ny::Int
        h::Float64
        x_p::Vector{Float64}
        y_p::Vector{Float64}
        x_u::Vector{Float64}
        y_v::Vector{Float64} 
    end

    struct LagrangianCircle
        X::Vector{Float64} 
        Y::Vector{Float64} 
        ds::Float64         # arc length per marker 
    end
    
    function horizontal_velocity_profile(Ny, U_max)
        return U_max * ones(Ny)  # uniform profile
    end

    function build_cylinder(xc, yc, R, h)
        N  = round(Int, 2π * R / h) 
        ds = 2π * R / N
        theta  = range(0, 2π, length = N + 1)[1:N]     # drop the duplicated endpoint
        X  = xc .+ R .* cos.(theta)
        Y  = yc .+ R .* sin.(theta)
        return LagrangianCircle(collect(X), collect(Y), ds)
    end

    function initial_condition(Nx, Ny, horizontal_velocity, cylinder_obj; domain_size=(1.0, 1.0))
        Lx, Ly = domain_size
        h = Lx / Nx

        x_p = collect(range(h/2, Lx - h/2, length=Nx))
        y_p = collect(range(h/2, Ly - h/2, length=Ny))
        
        x_u = collect(range(0, Lx, length=Nx+1))
        y_v = collect(range(0, Ly, length=Ny+1))
    
        grid = PhysicalGrid(Nx, Ny, h, x_p, y_p, x_u, y_v)
        
        # Create a staggered grid for velocity and pressure
        u = zeros(Nx + 1, Ny)
        v = zeros(Nx, Ny + 1)
        u[1, :] .= horizontal_velocity

        p = zeros(Nx, Ny)
        
        u_star = zeros(Nx + 1, Ny)
        v_star = zeros(Nx, Ny + 1)

        cylinder_lagrangian = build_cylinder(cylinder_obj[1], cylinder_obj[2], cylinder_obj[3], h)
    
        return grid, u, v, p, u_star, v_star, cylinder_lagrangian
    end
end


