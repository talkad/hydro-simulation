module plot_utils
    
    # using Plots
    using FileIO
    using CairoMakie
    using Interpolations

    export plot_streamlines, plot_horizontal_velocity, plot_vertical_velocity, plot_pressure
    

    function plot_streamlines(u, v, grid, step_num, t, cylinder_lagrangian=nothing)
        Nx, Ny = grid.Nx, grid.Ny

        uc = 0.5 .* (u[1:Nx, :] .+ u[2:Nx+1, :])
        vc = 0.5 .* (v[:, 1:Ny] .+ v[:, 2:Ny+1])

        speed = sqrt.(uc.^2 .+ vc.^2)          # scalar field to contour

        xr = range(grid.h/2, step=grid.h, length=Nx)   # matches your x_p exactly
        yr = range(grid.h/2, step=grid.h, length=Ny)

        itp_u = extrapolate(scale(interpolate(uc, BSpline(Linear())), xr, yr), Flat())
        itp_v = extrapolate(scale(interpolate(vc, BSpline(Linear())), xr, yr), Flat())

        vel(x) = Point2f(itp_u(x[1], x[2]), itp_v(x[1], x[2]))

        fig = Figure(size = (1000, 200))
        ax  = Axis(fig[1, 1]; title = "t = $(round(t, digits=3))",
                xlabel = "x", ylabel = "y")

        hm = contourf!(ax, xr, yr, speed; colormap = :viridis, levels = 15)

        contour!(ax, xr, yr, speed; color = (:white, 0.5), linewidth = 0.8, levels = 15)

        streamplot!(ax, vel, xr[1]..xr[end], yr[1]..yr[end];
                    colormap = :grays, density = 1.5,
                    linewidth = 1.2, arrow_size = 8)

        if cylinder_lagrangian !== nothing
            poly!(ax, Point2f.(cylinder_lagrangian.X, cylinder_lagrangian.Y);
                  color = :white)
        end

        Colorbar(fig[1, 2], hm, label = "|U|")
        limits!(ax, 0, grid.x_p[end] + grid.h/2, 0, grid.y_p[end] + grid.h/2)

        save("frames/stream_$(lpad(step_num, 5, '0')).png", fig)
        return fig
    end


    function plot_horizontal_velocity(u, grid, step_num, t, cylinder_lagrangian=nothing)
        fig = Figure(size = (1000, 200))
        ax  = Axis(fig[1, 1]; title = "u   (t = $(round(t, digits=3)))",
                xlabel = "x", ylabel = "y")

        hm = contourf!(ax, grid.x_u, grid.y_p, u; colormap = :balance, levels = 15)
        contour!(ax, grid.x_u, grid.y_p, u; color = (:black, 0.3), linewidth = 0.5, levels = 15)

        if cylinder_lagrangian !== nothing
            poly!(ax, Point2f.(cylinder_lagrangian.X, cylinder_lagrangian.Y); color = :white)
        end

        Colorbar(fig[1, 2], hm, label = "u")
        limits!(ax, 0, grid.x_p[end] + grid.h/2, 0, grid.y_p[end] + grid.h/2)

        save("frames/u_$(lpad(step_num, 5, '0')).png", fig)
        return fig
    end


    function plot_vertical_velocity(v, grid, step_num, t, cylinder_lagrangian=nothing)
        fig = Figure(size = (1000, 200))
        ax  = Axis(fig[1, 1]; title = "v   (t = $(round(t, digits=3)))",
                xlabel = "x", ylabel = "y")

        hm = contourf!(ax, grid.x_p, grid.y_v, v; colormap = :balance, levels = 15)
        contour!(ax, grid.x_p, grid.y_v, v; color = (:black, 0.3), linewidth = 0.5, levels = 15)

        if cylinder_lagrangian !== nothing
            poly!(ax, Point2f.(cylinder_lagrangian.X, cylinder_lagrangian.Y); color = :white)
        end

        Colorbar(fig[1, 2], hm, label = "v")
        limits!(ax, 0, grid.x_p[end] + grid.h/2, 0, grid.y_p[end] + grid.h/2)

        save("frames/v_$(lpad(step_num, 5, '0')).png", fig)
        return fig
    end


    function plot_pressure(p, grid, step_num, t, cylinder_lagrangian=nothing)
        fig = Figure(size = (1000, 200))
        ax  = Axis(fig[1, 1]; title = "p   (t = $(round(t, digits=3)))",
                xlabel = "x", ylabel = "y")

        hm = contourf!(ax, grid.x_p, grid.y_p, p; colormap = :balance, levels = 15)
        contour!(ax, grid.x_p, grid.y_p, p; color = (:black, 0.3), linewidth = 0.5, levels = 15)

        if cylinder_lagrangian !== nothing
            poly!(ax, Point2f.(cylinder_lagrangian.X, cylinder_lagrangian.Y); color = :white)
        end

        Colorbar(fig[1, 2], hm, label = "p")
        limits!(ax, 0, grid.x_p[end] + grid.h/2, 0, grid.y_p[end] + grid.h/2)

        save("frames/p_$(lpad(step_num, 5, '0')).png", fig)
        return fig
    end

end