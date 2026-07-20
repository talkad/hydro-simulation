module bdf2

    export intermediate_velocity


    # BDF2:
    #   (I - (2dt/3)*nu*L) * u* = (4/3)u^n - (1/3)u^{n-1}
    #                             - (2dt/3)*[conv(u^n) + grad(p^n)]
    function intermediate_velocity(u, u_prev, v, v_prev, p, ops, Nx, Ny, dt, F_u, F_v, F, Ru, Rv)
        u_vec      = u[:]
        u_prev_vec = u_prev[:]
        v_vec      = v[:]
        v_prev_vec = v_prev[:]
        p_vec      = p[:]
        
        # # Central differences for convective terms
        # v_on_u = ops.INTERP_V2U * v_vec
        # conv_u = u_vec .* (ops.Dx_u * u_vec) + v_on_u .* (ops.Dy_u * u_vec)

        # u_on_v = ops.INTERP_U2V * u_vec
        # conv_v = u_on_v .* (ops.Dx_v * v_vec) + v_vec .* (ops.Dy_v * v_vec)

        # upwind differences for convective terms
        v_on_u = ops.INTERP_V2U * v_vec
        conv_u = max.(u_vec, 0) .* (ops.Dx_u_b * u_vec) .+ min.(u_vec, 0) .* (ops.Dx_u_f * u_vec) .+
                max.(v_on_u, 0) .* (ops.Dy_u_b * u_vec) .+ min.(v_on_u, 0) .* (ops.Dy_u_f * u_vec)

        u_on_v = ops.INTERP_U2V * u_vec
        conv_v = max.(u_on_v, 0) .* (ops.Dx_v_b * v_vec) .+ min.(u_on_v, 0) .* (ops.Dx_v_f * v_vec) .+
         max.(v_vec, 0) .* (ops.Dy_v_b * v_vec) .+ min.(v_vec, 0) .* (ops.Dy_v_f * v_vec)

        dt2 = 2dt / 3
        rhs_u = (4/3) .* u_vec - (1/3) .* u_prev_vec - dt2 .* (conv_u +  ops.GRAD_X * p_vec - Ru * F[:, 1])
        rhs_v = (4/3) .* v_vec - (1/3) .* v_prev_vec - dt2 .* (conv_v + ops.GRAD_Y * p_vec - Rv * F[:, 2])

        u_star_vec = F_u \ rhs_u
        v_star_vec = F_v \ rhs_v

        return reshape(u_star_vec, Nx+1, Ny), reshape(v_star_vec, Nx, Ny+1)
    end

end
