module bc

    export apply_vel_derichlet!

    function apply_vel_derichlet!(u, v, horizontal_velocity)
        u[:, end] .= 0.0
        u[:, 1] .= 0.0
        u[1, :] .= horizontal_velocity
        u[end, :] .= u[end-1, :]

        v[1, :] .= 0.0
        v[end, :] .= v[end-1, :]  
        v[:, 1] .= 0.0
        v[:, end] .= 0.0
    end

    
end