module bc

    export apply_vel_derichlet!

    function apply_vel_derichlet!(u, v, U_lid)
        u[:, end] .= U_lid
        u[:, 1] .= 0.0
        u[1, :] .= 0.0
        u[end, :] .= 0.0

        v[1, :] .= 0.0
        v[end, :] .= 0.0  
        v[:, 1] .= 0.0    
        v[:, end] .= 0.0   

    end


end