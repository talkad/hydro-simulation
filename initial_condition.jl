module ic

    export initial_condition

    struct PhysicalGrid
        Nx::Int
        Ny::Int
        dx::Float64
        dy::Float64
        x_p::Vector{Float64}
        y_p::Vector{Float64}
        x_u::Vector{Float64}
        y_v::Vector{Float64} 
    end
    
    function initial_condition(Nx, Ny; domain_size=(1.0, 1.0), U_lid=1.0)
        Lx, Ly = domain_size
        dx = Lx / Nx
        dy = Ly / Ny

        x_p = collect(range(dx/2, Lx - dx/2, length=Nx))
        y_p = collect(range(dy/2, Ly - dy/2, length=Ny))
        
        x_u = collect(range(0, Lx, length=Nx+1))
        y_v = collect(range(0, Ly, length=Ny+1))
    
        grid = PhysicalGrid(Nx, Ny, dx, dy, x_p, y_p, x_u, y_v)
        
        # Create a staggered grid for velocity and pressure
        u = zeros(Nx + 1, Ny)
        v = zeros(Nx, Ny + 1)
        u[:, Ny] .= U_lid
        p = zeros(Nx, Ny)
        
        u_star = zeros(Nx + 1, Ny)
        v_star = zeros(Nx, Ny + 1)
    
        return grid, u, v, p, u_star, v_star
    end
end


