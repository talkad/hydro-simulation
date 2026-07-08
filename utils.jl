module utils

    using SparseArrays, LinearAlgebra, SuiteSparse
    export Operators

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

    function Operators(Nx, Ny, dx, dy)
        D1(n, h) = spdiagm(-1 => fill(-1/(2h), n-1), 1 => fill(1/(2h), n-1))
        L1(n, h) = spdiagm(-1 => fill(1/h^2, n-1), 0 => fill(-2/h^2, n), 1 => fill(1/h^2, n-1))
        A1(n)    = spdiagm(0 => fill(0.5, n), 1 => fill(0.5, n-1))

        Dx_u = kron(I(Ny),      D1(Nx+1, dx))
        Dy_u = kron(D1(Ny, dy), I(Nx+1))
        L_u  = kron(I(Ny),      L1(Nx+1, dx)) + kron(L1(Ny,    dy), I(Nx+1))

        Dx_v = kron(I(Ny+1),      D1(Nx, dx))
        Dy_v = kron(D1(Ny+1, dy), I(Nx))
        L_v  = kron(I(Ny+1),      L1(Nx, dx)) + kron(L1(Ny+1,  dy), I(Nx))

        interp_v2u = kron(A1(Ny+1)[1:Ny, 1:Ny+1],
                          spdiagm(0 => fill(0.5, Nx), -1 => fill(0.5, Nx))[1:Nx+1, 1:Nx])
        interp_u2v = kron(spdiagm(0 => fill(0.5, Ny), -1 => fill(0.5, Ny))[1:Ny+1, 1:Ny],
                          A1(Nx+1)[1:Nx, 1:Nx+1])

        # Upwind
        Dfwd(n, h)  = spdiagm(0 => fill(-1/h, n),  1 => fill(1/h, n-1))  
        Dback(n, h) = spdiagm(0 => fill( 1/h, n), -1 => fill(-1/h, n-1)) 

        Dx_u_f = kron(I(Ny),        Dfwd(Nx+1, dx));  Dx_u_b = kron(I(Ny),         Dback(Nx+1, dx))
        Dy_u_f = kron(Dfwd(Ny, dy), I(Nx+1));         Dy_u_b = kron(Dback(Ny, dy), I(Nx+1))
        Dx_v_f = kron(I(Ny+1),      Dfwd(Nx, dx));    Dx_v_b = kron(I(Ny+1),       Dback(Nx, dx))
        Dy_v_f = kron(Dfwd(Ny+1,dy),I(Nx));           Dy_v_b = kron(Dback(Ny+1,dy),I(Nx))

        # --- pressure operators, built as a matched pair ---
        # gradient: cells -> faces, (p_i - p_{i-1})/dx at interior faces
        Dx_p2u = spdiagm(0 => fill(1/dx, Nx), -1 => fill(-1/dx, Nx))[1:Nx+1, 1:Nx]
        Dy_p2v = spdiagm(0 => fill(1/dy, Ny), -1 => fill(-1/dy, Ny))[1:Ny+1, 1:Ny]

        # Neumann: no pressure gradient at wall faces -> zero the wall-face rows
        Dx_p2u[1, :] .= 0; Dx_p2u[end, :] .= 0
        Dy_p2v[1, :] .= 0; Dy_p2v[end, :] .= 0

        GRAD_X = kron(I(Ny),  Dx_p2u)
        GRAD_Y = kron(Dy_p2v, I(Nx))

        # divergence: faces -> cells, defined as -GRADᵀ so that L = DIV*GRAD is SPD-consistent
        DIV_X = -GRAD_X'
        DIV_Y = -GRAD_Y'

        L_p = DIV_X*GRAD_X + DIV_Y*GRAD_Y   # exact Neumann Laplacian for THIS grad/div

        Lp_factorized = copy(L_p)
        Lp_factorized[1, :] .= 0.0; Lp_factorized[1, 1] = 1.0     # system is singular
        Lp_factorized = lu(Lp_factorized)

        return Operators(Dx_u, Dy_u, L_u, Dx_v, Dy_v, L_v,
                         interp_v2u, interp_u2v,
                         GRAD_X, GRAD_Y, DIV_X, DIV_Y, L_p, Lp_factorized,
                         Dx_u_f, Dx_u_b, Dy_u_f, Dy_u_b,
                         Dx_v_f, Dx_v_b, Dy_v_f, Dy_v_b)
    end

end