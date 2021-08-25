using SparseArrays


function conjugate_gradient(product_function,
                            edges_list::Array{<:Pair{<:Integer, <:Integer}, 1},
                            E::SparseMatrixCSC{<:Real, <:Integer},
                            D::Array{<:Real, 1},
                            b::Array{<:Real, 1},
                            tol::Real,
                            max_iter::Integer)
    if max_iter < 1
        println("ERROR: max_iter should be > 0")
        return Nothing, Nothing
    end

    n = size(b, 1)
    x = zeros(n)
    r_old = b
    d = b

    D_inv = ones(size(D, 1)) ./ D
    tol *= norm(b)

    residuals = []

    for _ in 1:max_iter
        rr_old = r_old'r_old

        if sqrt(rr_old) <= tol
            println("Early stopping: ", length(residuals))
            return x, residuals
        end
        
        # TODO: pass as argument
        Ld = product_function(edges_list, E, D_inv, d)

        α = rr_old / (d'Ld)

        # TODO: explain
        #if α <= eps()
        #    println("Error. α=", α)
            #return x, residuals
        #end

        x = x + α * d
        r = r_old - α * Ld
        β = (r'r) / rr_old
        d = r + β * d

        r_old = r
        append!(residuals, norm(r))
    end

    return x, residuals
end
