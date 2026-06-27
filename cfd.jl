module cfd

    using SparseArrays, LinearAlgebra, SuiteSparse

    export intermediate_velocity, solve_pressure, update_velocity


    # Define operators
    # Central difference 1D in space
    D1(n, h) = spdiagm(-1 => fill(-1/(2h), n-1), 1 => fill(1/(2h), n-1))
    # Laplacian 1D in space
    L1(n, h) = spdiagm(-1 => fill(1/h^2, n-1), 0 => fill(-2/h^2, n), 1 => fill(1/h^2, n-1))
    # Averaging 1D
    A1(n) = spdiagm(0 => fill(0.5, n), 1 => fill(0.5, n-1))

    #Interpolation V to U: (Nx * Ny+1) -> (Nx+1 * Ny)
    INTERP_V2U(Nx, Ny) = kron(A1(Ny+1)[1:Ny, 1:Ny+1], spdiagm(0 => fill(0.5, Nx), -1 => fill(0.5, Nx))[1:Nx+1, 1:Nx])

    #Interpolation U to V: (Nx+1 * Ny) -> (Nx * Ny+1)
    INTERP_U2V(Nx, Ny) = kron(spdiagm(0 => fill(0.5, Ny), -1 => fill(0.5, Ny))[1:Ny+1, 1:Ny], A1(Nx+1)[1:Nx, 1:Nx+1])


    function intermediate_velocity(u, v, Nx, Ny, dx, dy, dt, nu)
        Dx_u = kron(I(Ny), D1(Nx+1, dx))
        Dy_u = kron(D1(Ny, dy), I(Nx+1))
        L_u  = kron(I(Ny), L1(Nx+1, dx)) + kron(L1(Ny, dy), I(Nx+1))
    
        Dx_v = kron(I(Ny+1), D1(Nx, dx))
        Dy_v = kron(D1(Ny+1, dy), I(Nx))
        L_v  = kron(I(Ny+1), L1(Nx, dx)) + kron(L1(Ny+1, dy), I(Nx))

        # Compute Terms ---
        u_vec = u[:]
        v_vec = v[:]
    
        # Convection: u*du/dx + v_avg*du/dy
        v_on_u = INTERP_V2U(Nx, Ny) * v_vec
        conv_u = u_vec .* (Dx_u * u_vec) + v_on_u .* (Dy_u * u_vec)
    
        # Convection: u_avg*dv/dx + v*dv/dy
        u_on_v = INTERP_U2V(Nx, Ny) * u_vec
        conv_v = u_on_v .* (Dx_v * v_vec) + v_vec .* (Dy_v * v_vec)
    
        # Diffusion: nu * Laplacian
        diff_u = nu * L_u * u_vec
        diff_v = nu * L_v * v_vec
    
        # Time Advancement (Predictor) ---
        u_star_vec = u_vec + dt * (-conv_u + diff_u)
        v_star_vec = v_vec + dt * (-conv_v + diff_v)
    
        return reshape(u_star_vec, Nx+1, Ny), reshape(v_star_vec, Nx, Ny+1)
    end


    function solve_pressure(u_star, v_star, Nx, Ny, dx, dy, rho, dt)
        Dx = spdiagm(0 => fill(-1/dx, Nx), 1 => fill(1/dx, Nx))[1:Nx, :]
        Dy = spdiagm(0 => fill(-1/dy, Ny), 1 => fill(1/dy, Ny))[1:Ny, :]
        
        DIV_X = kron(I(Ny), Dx)
        DIV_Y = kron(Dy, I(Nx))
        
        L = -(DIV_X * (DIV_X') + DIV_Y * (DIV_Y'))

        b = (rho / dt) * (DIV_X * u_star[:] + DIV_Y * v_star[:])

        L[1, :] .= 0.0
        L[1, 1] = 1.0
        b[1] = 0.0

        p_vec = L \ b
        
        return reshape(p_vec, Nx, Ny)
    end

    function update_velocity(u_star, v_star, p, Nx, Ny, dx, dy, rho, dt)
        Dx_p2u = spdiagm(0 => fill(1/dx, Nx), -1 => fill(-1/dx, Nx))[1:Nx+1, 1:Nx]
        Dy_p2v = spdiagm(0 => fill(1/dy, Ny), -1 => fill(-1/dy, Ny))[1:Ny+1, 1:Ny]
    
        GRAD_X = kron(I(Ny), Dx_p2u)
        GRAD_Y = kron(Dy_p2v, I(Nx))
    
        u_new_vec = u_star[:] - (dt / rho) * (GRAD_X * p[:])
        v_new_vec = v_star[:] - (dt / rho) * (GRAD_Y * p[:])
    
        return reshape(u_new_vec, Nx+1, Ny), reshape(v_new_vec, Nx, Ny+1)
    end
end