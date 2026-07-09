module simple

    using SparseArrays, LinearAlgebra, SuiteSparse

    export solver

    function solver(u_star, v_star, p, ops, grid, rho, dt, F_u, F_v;
                    u_prev=nothing, v_prev=nothing, tol=1e-8, max_iter=50)
        Nx, Ny = grid.Nx, grid.Ny
        F = ops.Lp_factorized

        u_vec = u_star[:]
        v_vec = v_star[:]
        p_vec = p[:]

        for _ in 1:max_iter
            # pressure projection
            b = (rho/dt) * (ops.DIV_X*u_vec + ops.DIV_Y*v_vec); b[1] = 0.0
            pc = F \ b

            u_vec .-= (dt/rho) * (ops.GRAD_X * pc)
            v_vec .-= (dt/rho) * (ops.GRAD_Y * pc)
            p_vec .+= pc

            maximum(abs, ops.DIV_X*u_vec + ops.DIV_Y*v_vec) < tol && break

            # re-predict intermediate velocity with the UPDATED pressure
            u_mat = reshape(u_vec, Nx+1, Ny)
            v_mat = reshape(v_vec, Nx,   Ny+1)
            p_mat = reshape(p_vec, Nx,   Ny)

            if u_prev === nothing || v_prev === nothing
                u_new, v_new = first_order.intermediate_velocity(
                    u_mat, v_mat, p_mat, ops, Nx, Ny, dt, rho, F_u, F_v)
            else
                u_new, v_new = bdf2.intermediate_velocity(
                    u_mat, u_prev, v_mat, v_prev, p_mat, ops, Nx, Ny, dt, rho, F_u, F_v)
            end

            u_vec = u_new[:]
            v_vec = v_new[:]
        end

        return reshape(u_vec, Nx+1, Ny), reshape(v_vec, Nx, Ny+1), reshape(p_vec, Nx, Ny)
    end

end
