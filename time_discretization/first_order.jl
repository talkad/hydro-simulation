module first_order

    export intermediate_velocity


    # Backward Euler
    #   (I - dt*nu*L) * u* = u^n - dt*conv(u^n) - (dt/rho)*grad(p^n)
    function intermediate_velocity(u, v, p, ops, Nx, Ny, dt, rho, F_u, F_v)
        u_vec = u[:]
        v_vec = v[:]
        p_vec = p[:] 

        v_on_u = ops.INTERP_V2U * v_vec
        conv_u = u_vec .* (ops.Dx_u * u_vec) + v_on_u .* (ops.Dy_u * u_vec)

        u_on_v = ops.INTERP_U2V * u_vec
        conv_v = u_on_v .* (ops.Dx_v * v_vec) + v_vec .* (ops.Dy_v * v_vec)

        # # upwind differences for convective terms
        # v_on_u = ops.INTERP_V2U * v_vec
        # conv_u = max.(u_vec, 0) .* (ops.Dx_u_b * u_vec) .+ min.(u_vec, 0) .* (ops.Dx_u_f * u_vec) .+
        #         max.(v_on_u, 0) .* (ops.Dy_u_b * u_vec) .+ min.(v_on_u, 0) .* (ops.Dy_u_f * u_vec)

        # u_on_v = ops.INTERP_U2V * u_vec
        # conv_v = max.(u_on_v, 0) .* (ops.Dx_v_b * v_vec) .+ min.(u_on_v, 0) .* (ops.Dx_v_f * v_vec) .+
        #  max.(v_vec, 0) .* (ops.Dy_v_b * v_vec) .+ min.(v_vec, 0) .* (ops.Dy_v_f * v_vec)

        rhs_u = u_vec - dt * conv_u - (dt / rho) * (ops.GRAD_X * p_vec)
        rhs_v = v_vec - dt * conv_v - (dt / rho) * (ops.GRAD_Y * p_vec)

        u_star_vec = F_u \ rhs_u
        v_star_vec = F_v \ rhs_v

        return reshape(u_star_vec, Nx+1, Ny), reshape(v_star_vec, Nx, Ny+1)
    end

end
