module utils

    using SparseArrays, LinearAlgebra, SuiteSparse
    export build_operators, ib_operators

    struct Operators
        Dx_u::SparseMatrixCSC{Float64, Int}
        Dy_u::SparseMatrixCSC{Float64, Int}
        L_u::SparseMatrixCSC{Float64, Int}
        Dx_v::SparseMatrixCSC{Float64, Int}
        Dy_v::SparseMatrixCSC{Float64, Int}
        L_v::SparseMatrixCSC{Float64, Int}
        INTERP_V2U::SparseMatrixCSC{Float64, Int}
        INTERP_U2V::SparseMatrixCSC{Float64, Int}
        GRAD_X::SparseMatrixCSC{Float64, Int}
        GRAD_Y::SparseMatrixCSC{Float64, Int}
        DIV_X::SparseMatrixCSC{Float64, Int}   # u-faces -> cells
        DIV_Y::SparseMatrixCSC{Float64, Int}   # v-faces -> cells
        DIV_X_bc::SparseMatrixCSC{Float64, Int}  # missing x=0/x=Lx boundary-face flux, see build_operators
        L_p::SparseMatrixCSC{Float64, Int}     # Neumann pressure Laplacian = DIV*GRAD
        Lp_factorized::SparseArrays.UMFPACK.UmfpackLU{Float64, Int}
        Dx_u_f::SparseMatrixCSC{Float64, Int}
        Dx_u_b::SparseMatrixCSC{Float64, Int}
        Dy_u_f::SparseMatrixCSC{Float64, Int}
        Dy_u_b::SparseMatrixCSC{Float64, Int}
        Dx_v_f::SparseMatrixCSC{Float64, Int}
        Dx_v_b::SparseMatrixCSC{Float64, Int}
        Dy_v_f::SparseMatrixCSC{Float64, Int}
        Dy_v_b::SparseMatrixCSC{Float64, Int}
    end

    function build_operators(Nx, Ny, dh)
        D1(n, h) = spdiagm(-1 => fill(-1/(2h), n-1), 1 => fill(1/(2h), n-1))
        L1(n, h) = spdiagm(-1 => fill(1/h^2, n-1), 0 => fill(-2/h^2, n), 1 => fill(1/h^2, n-1))
        A1(n)    = spdiagm(0 => fill(0.5, n), 1 => fill(0.5, n-1))

        Dx_u = kron(I(Ny),      D1(Nx+1, dh))
        Dy_u = kron(D1(Ny, dh), I(Nx+1))
        L_u  = kron(I(Ny),      L1(Nx+1, dh)) + kron(L1(Ny,    dh), I(Nx+1))

        Dx_v = kron(I(Ny+1),      D1(Nx, dh))
        Dy_v = kron(D1(Ny+1, dh), I(Nx))
        L_v  = kron(I(Ny+1),      L1(Nx, dh)) + kron(L1(Ny+1,  dh), I(Nx))

        interp_v2u = kron(A1(Ny+1)[1:Ny, 1:Ny+1],
                          spdiagm(0 => fill(0.5, Nx), -1 => fill(0.5, Nx))[1:Nx+1, 1:Nx])
        interp_u2v = kron(spdiagm(0 => fill(0.5, Ny), -1 => fill(0.5, Ny))[1:Ny+1, 1:Ny],
                          A1(Nx+1)[1:Nx, 1:Nx+1])

        # Upwind
        Dfwd(n, h)  = spdiagm(0 => fill(-1/h, n),  1 => fill(1/h, n-1))  
        Dback(n, h) = spdiagm(0 => fill( 1/h, n), -1 => fill(-1/h, n-1)) 

        Dx_u_f = kron(I(Ny),        Dfwd(Nx+1, dh));  Dx_u_b = kron(I(Ny),         Dback(Nx+1, dh))
        Dy_u_f = kron(Dfwd(Ny, dh), I(Nx+1));         Dy_u_b = kron(Dback(Ny, dh), I(Nx+1))
        Dx_v_f = kron(I(Ny+1),      Dfwd(Nx, dh));    Dx_v_b = kron(I(Ny+1),       Dback(Nx, dh))
        Dy_v_f = kron(Dfwd(Ny+1,dh),I(Nx));           Dy_v_b = kron(Dback(Ny+1,dh),I(Nx))

        # --- pressure operators, built as a matched pair ---
        # gradient: cells -> faces, (p_i - p_{i-1})/dx at interior faces
        Dx_p2u = spdiagm(0 => fill(1/dh, Nx), -1 => fill(-1/dh, Nx))[1:Nx+1, 1:Nx]
        Dy_p2v = spdiagm(0 => fill(1/dh, Ny), -1 => fill(-1/dh, Ny))[1:Ny+1, 1:Ny]

        # Neumann: no pressure gradient at wall faces -> zero the wall-face rows
        Dx_p2u[1, :] .= 0; Dx_p2u[end, :] .= 0
        Dy_p2v[1, :] .= 0; Dy_p2v[end, :] .= 0

        GRAD_X = kron(I(Ny),  Dx_p2u)
        GRAD_Y = kron(Dy_p2v, I(Nx))

        # divergence: faces -> cells, defined as -GRADᵀ so that L = DIV*GRAD is SPD-consistent
        DIV_X = -GRAD_X'
        DIV_Y = -GRAD_Y'

        Dx_p2u_full = spdiagm(0 => fill(1/dh, Nx), -1 => fill(-1/dh, Nx))[1:Nx+1, 1:Nx]
        GRAD_X_full = kron(I(Ny), Dx_p2u_full)
        DIV_X_bc    = -GRAD_X_full' - DIV_X

        L_p = DIV_X*GRAD_X + DIV_Y*GRAD_Y   # exact Neumann Laplacian for THIS grad/div

        Lp_factorized = copy(L_p)
        Lp_factorized[1, :] .= 0.0; Lp_factorized[1, 1] = 1.0     # system is singular
        Lp_factorized = lu(Lp_factorized)

        return Operators(Dx_u, Dy_u, L_u, Dx_v, Dy_v, L_v,
                         interp_v2u, interp_u2v,
                         GRAD_X, GRAD_Y, DIV_X, DIV_Y, DIV_X_bc, L_p, Lp_factorized,
                         Dx_u_f, Dx_u_b, Dy_u_f, Dy_u_b,
                         Dx_v_f, Dx_v_b, Dy_v_f, Dy_v_b)
    end


    @inline function roma_phi(r::Real)
        a = abs(r)
        if a <= 0.5
            return (1 + sqrt(1 - 3a^2)) / 3
        elseif a <= 1.5
            return (5 - 3a - sqrt(1 - 3*(1 - a)^2)) / 6
        else
            return zero(float(r))
        end
    end

    function build_IR(Xk, Yk, ds, xs, ys, h)
        NL = length(Xk)
        nx, ny = length(xs), length(ys)
        lin = LinearIndices((nx, ny))
        rows = Int[]; cols = Int[]; vals = Float64[]

        for k in 1:NL
            i0 = searchsortedfirst(xs, Xk[k] - 1.5h)
            i1 = searchsortedlast(xs,  Xk[k] + 1.5h)
            j0 = searchsortedfirst(ys, Yk[k] - 1.5h)
            j1 = searchsortedlast(ys,  Yk[k] + 1.5h)

            for j in j0:j1, i in i0:i1
                w = roma_phi((xs[i] - Xk[k]) / h) * roma_phi((ys[j] - Yk[k]) / h)
                if w != 0
                    push!(rows, k); push!(cols, lin[i, j]); push!(vals, w)
                end
            end
        end

        Iop = sparse(rows, cols, vals, NL, nx*ny)          # interpolation
        Rop = sparse(Iop') * Diagonal(ds ./ h)           # regularization (adjoint)
        return Iop, Rop
    end

    function ib_operators(cylinder_lagrangian, grid)
        Xk, Yk = cylinder_lagrangian.X, cylinder_lagrangian.Y
        ds = fill(cylinder_lagrangian.ds, length(Xk))
        h  = grid.h

        print("Building IB operators for $(length(Xk)) markers, ds=$(maximum(ds)), h=$h\n")

        xu, yu = grid.x_u, grid.y_p   # u-faces
        xv, yv = grid.x_p, grid.y_v   # v-faces

        Iu, Ru = build_IR(Xk, Yk, ds, xu, yu, h)   # x-component, on u-faces
        Iv, Rv = build_IR(Xk, Yk, ds, xv, yv, h)   # y-component, on v-faces
        return  Iu, Ru, Iv, Rv
    end

 
end