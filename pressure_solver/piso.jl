module piso

    using SparseArrays, LinearAlgebra, SuiteSparse
    export solver

    function solver(u_star, v_star, p, ops, grid, rho, dt, A_u, A_v; n_corr=2)
        Nx, Ny = grid.Nx, grid.Ny

        Du = diag(A_u);           Dv = diag(A_v)
        Ou = A_u - Diagonal(Du);  Ov = A_v - Diagonal(Dv)
        Dinv_u = 1.0 ./ Du;       Dinv_v = 1.0 ./ Dv

        GXw = Diagonal(Dinv_u) * ops.GRAD_X
        GYw = Diagonal(Dinv_v) * ops.GRAD_Y

        Lhat = ops.DIV_X * GXw + ops.DIV_Y * GYw
        Lhat = copy(Lhat)
        Lhat[1, :] .= 0.0; Lhat[1, 1] = 1.0
        F = lu(Lhat)

        us = u_star[:]; vs = v_star[:]           
        uc = zeros(length(us)); vc = zeros(length(vs))  
        hu = zeros(length(us)); hv = zeros(length(vs)) 
        pc = zeros(Nx*Ny)

        for it in 1:n_corr
            if it > 1                          
                hu = -Dinv_u .* (Ou * uc)
                hv = -Dinv_v .* (Ov * vc)
            end

            b = (rho/dt) * (ops.DIV_X * (us .+ hu) + ops.DIV_Y * (vs .+ hv))
            b[1] = 0.0
            pc = F \ b

            uc = hu .- (dt/rho) * (GXw * pc)  
            vc = hv .- (dt/rho) * (GYw * pc)
        end

        u = us .+ uc
        v = vs .+ vc
        p_vec = p[:] .+ pc

        return reshape(u, Nx+1, Ny), reshape(v, Nx, Ny+1), reshape(p_vec, Nx, Ny)
    end

end