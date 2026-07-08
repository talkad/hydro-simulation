module simple

    using SparseArrays, LinearAlgebra, SuiteSparse

    export solver

    function solver(u_star, v_star, p, ops, grid, rho, dt; A_u=nothing, A_v=nothing, tol=1e-8, max_iter=50)
        Nx, Ny = grid.Nx, grid.Ny
        F = ops.Lp_factorized
        
        u_vec = u_star[:]
        v_vec = v_star[:]
        p_vec = p[:]

        for _ in 1:max_iter
            b = (rho/dt) * (ops.DIV_X*u_vec + ops.DIV_Y*v_vec); b[1] = 0.0
            pc = F \ b

            u_vec .-= (dt/rho) * (ops.GRAD_X * pc)
            v_vec .-= (dt/rho) * (ops.GRAD_Y * pc)
            p_vec .+= pc

            maximum(abs, ops.DIV_X*u_vec + ops.DIV_Y*v_vec) < tol && break
        end

        return reshape(u_vec, Nx+1, Ny), reshape(v_vec, Nx, Ny+1), reshape(p_vec, Nx, Ny)
    end

end
