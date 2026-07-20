module simple

    using SparseArrays, LinearAlgebra, IterativeSolvers
    export simple_step, ib_setup

    function build_blocks(ops, Iu, Ru, Iv, Rv)
        NL = size(Iu, 1)
        L  = -ops.L_p
        Bt = [ops.DIV_X*Ru   ops.DIV_Y*Rv]              # Nc x 2NL
        B  = [-(Iu*ops.DIV_X');                         # 2NL x Nc   G = -D'
            -(Iv*ops.DIV_Y')]
        C  = sparse(-0.5I, 2NL, 2NL)                    # 2NL, not NL
        return L, B, Bt, C
    end

    
    function make_poisson_solver(L::SparseMatrixCSC; pin::Int=1)
        Lp = copy(L)
        Lp[pin, :] .= 0.0
        Lp[pin, pin] = 1.0
        F = lu(Lp)

        return function (b::AbstractVector)
            c = copy(b)
            c .-= sum(c) / length(c)        # remove constant  <-- must match rhs_p[pin]=0
            c[pin] = 0.0
            return F \ c
        end
    end

    function build_schur(B, Bt, C, psolve; thresh::Float64=1e-4, verbose::Bool=true)
        n = size(Bt, 2)                                  
        S = zeros(size(B, 1), n)
        Threads.@threads for j in 1:n
            z       = psolve(Vector(Bt[:, j]))           # z = L⁻¹ B'[:,j]
            S[:, j] = B * z                              # S[:,j] = B L⁻¹ B'[:,j]
        end
        S .-= Matrix(C)                                  # S = B L⁻¹ B' - C

        if thresh > 0
            S[abs.(S) .< thresh] .= 0.0
            verbose && println("  sparsified @ $thresh -> density $(round(count(!iszero,S)/length(S), digits=4))")
            return sparse(S)
        end
        return S
    end

    struct IBSchur
        L; B; Bt; C
        S
        psolve
        NL::Int
        Nc::Int
    end

    function ib_setup(ops, Iu, Ru, Iv, Rv; thresh=1e-3, pin=1, verbose=true)
        L, B, Bt, C = build_blocks(ops, Iu, Ru, Iv, Rv)
        psolve = make_poisson_solver(L; pin=pin)          # factor L once
        S      = build_schur(B, Bt, C, psolve; thresh=thresh, verbose=verbose)
        return IBSchur(L, B, Bt, C, S, psolve, size(Iu,1), size(L,1))
    end


    """
        schur_step(ib, rhs_p, rhs_F; use_direct=false)

        y  = L⁻¹ rhs_p                        Laplace solve #1
        F' = S \\ (B y - rhs_F)                small solve  (BiCgStab, <=3 iters)
        p' = L⁻¹(rhs_p - B'F')                Laplace solve #2
    """
    function schur_step(ib::IBSchur, rhs_p, rhs_F; tol=1e-9, use_direct=false, verbose=false)
        y   = ib.psolve(rhs_p)                            # Laplace #1
        rhs = ib.B * y .- rhs_F
        Fp  = if use_direct
            ib.S \ rhs
        else
            log = bicgstabl(ib.S, rhs; reltol=tol, log=true)
            verbose && println("  BiCgStab iters: $(log[2].iters)")
            log[1]
        end
        pp = ib.psolve(rhs_p .- ib.Bt * Fp)               # Laplace #2
        return pp, Fp
    end


    function simple_step(ib::IBSchur, ops, Iu, Ru, Iv, Rv, u_star, v_star, dt;
                            u_body=nothing, v_body=nothing, use_direct=false, verbose=true)
        NL = ib.NL
        Ub = u_body === nothing ? zeros(NL) : u_body
        Vb = v_body === nothing ? zeros(NL) : v_body
        
        u_shape = size(u_star)
        v_shape = size(v_star)

        u_star = u_star[:]
        v_star = v_star[:]

        rhs_p = -(3/(2dt)) .* (ops.DIV_X*u_star .+ ops.DIV_Y*v_star .+ ops.DIV_X_bc*u_star)
        rhs_F =  (3/(2dt)) .* vcat(Iu*u_star .- Ub, Iv*v_star .- Vb)

        pp, Fp = schur_step(ib, rhs_p, rhs_F; use_direct=use_direct, verbose=verbose)
        Fx, Fy = Fp[1:NL], Fp[NL+1:end]

        # --- velocity correction, Eq. (8):  u' = -(2/3)Δt(∇p' - R[F']),  ∇p' = -D'p'
        up = -(2/3)*dt .* (-(ops.DIV_X'*pp) .- Ru*Fx)
        vp = -(2/3)*dt .* (-(ops.DIV_Y'*pp) .- Rv*Fy)
        un = u_star .+ up
        vn = v_star .+ vp

        if verbose
            div  = maximum(abs, ops.DIV_X*un .+ ops.DIV_Y*vn)
            slip = max(maximum(abs, Iu*un .- Ub), maximum(abs, Iv*vn .- Vb))
            println("  div  = $(round(div,  sigdigits=3))    [expect ~1e-12]")
            println("  slip = $(round(slip, sigdigits=3))    [small, NOT machine-zero: C=-½I is approximate]")
        end

        un = reshape(un, u_shape)
        vn = reshape(vn, v_shape)

        return un, vn, pp, Fx, Fy
    end

end