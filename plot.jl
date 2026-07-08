module plot_utils
    
    # using Plots
    using FileIO
    using CairoMakie
    using Interpolations

    export plot_velocity_magnitude, plot_streamlines

    function plot_velocity_magnitude(u, v, grid, step, t)
        u_c = 0.5 .* (u[1:grid.Nx, :] .+ u[2:grid.Nx+1, :])
        v_c = 0.5 .* (v[:, 1:grid.Ny] .+ v[:, 2:grid.Ny+1])
        
        vel_mag = sqrt.(u_c.^2 .+ v_c.^2)

        p_heat = heatmap(grid.x_p, grid.y_p, vel_mag', 
                        c = :viridis, 
                        title = "Velocity Magnitude (Step $step, t=$(round(t, digits=3)))",
                        aspect_ratio = 1,
                        xlims = (0, 1), ylims = (0, 1))
        
        filename = "frames/frame_$(lpad(step, 5, "0")).png"
        savefig(p_heat, filename)
    end
    
    

    function plot_streamlines(u, v, grid, step_num, t)
        Nx, Ny = grid.Nx, grid.Ny

        uc = 0.5 .* (u[1:Nx, :] .+ u[2:Nx+1, :])
        vc = 0.5 .* (v[:, 1:Ny] .+ v[:, 2:Ny+1])

        speed = sqrt.(uc.^2 .+ vc.^2)          # scalar field to contour

        xr = range(grid.dx/2, step=grid.dx, length=Nx)   # matches your x_p exactly
        yr = range(grid.dy/2, step=grid.dy, length=Ny)

        itp_u = extrapolate(scale(interpolate(uc, BSpline(Linear())), xr, yr), Flat())
        itp_v = extrapolate(scale(interpolate(vc, BSpline(Linear())), xr, yr), Flat())

        vel(x) = Point2f(itp_u(x[1], x[2]), itp_v(x[1], x[2]))

        fig = Figure(size = (640, 600))
        ax  = Axis(fig[1, 1]; aspect = 1, title = "t = $(round(t, digits=3))",
                xlabel = "x", ylabel = "y")

        hm = contourf!(ax, xr, yr, speed; colormap = :viridis, levels = 15)

        contour!(ax, xr, yr, speed; color = (:white, 0.5), linewidth = 0.8, levels = 15)

        streamplot!(ax, vel, xr[1]..xr[end], yr[1]..yr[end];
                    colormap = :grays, density = 1.5,
                    linewidth = 1.2, arrow_size = 8)

        Colorbar(fig[1, 2], hm, label = "|U|")
        limits!(ax, 0, grid.x_p[end] + grid.dx/2, 0, grid.y_p[end] + grid.dy/2)

        save("frames/stream_$(lpad(step_num, 5, '0')).png", fig)
        return fig
    end

end