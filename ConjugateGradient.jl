using SparseArrays


function conjugate_gradient(product_function,
                            edges_list::Array{<:Pair{<:Integer, <:Integer}, 1},
                            E::SparseMatrixCSC{<:Real, <:Integer},
                            D::Array{<:Real, 1},
                            b::Array{<:Real, 1},
                            tol::Real,
                            max_iter::Integer)
    if max_iter <= 0
        println("ERROR: max_iter should be > 0")
        return Nothing, Nothing, Nothing, "error"
    end

    if tol < 0
        println("ERROR: tol should be ≥ 0")
        return Nothing, Nothing, Nothing, "error"
    end

    n = size(b, 1)
    x = zeros(n)
    r_old = b
    d = b

    D_inv = ones(size(D, 1)) ./ D  # D^(-1)
    tol *= norm(b)

    xs = []  # Value of x at each iteration
    residuals = []

    for _ in 1:max_iter
        rr_old = r_old'r_old

        if sqrt(rr_old) <= tol
            return x, xs, residuals, "converged"
        end
        
        # Ld = EDE^Td
        Ld = product_function(edges_list, E, D_inv, d)

        α = rr_old / (d'Ld)
        x .+= α .* d
        r = r_old .- α .* Ld
        β = (r'r) / rr_old
        d = r .+ β .* d

        r_old = r
        push!(xs, x)
        push!(residuals, norm(r))
    end

    return x, xs, residuals, "stopped"
end
