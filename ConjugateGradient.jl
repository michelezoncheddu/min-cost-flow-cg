using SparseArrays


function first_part_product(edges_list, E, D, b)
    x = zeros(size(E, 2))
    i = 1
    for (src, dst) in edges_list
        x[i] += D[i] * b[src]
        x[i] -= D[i] * b[dst]
        i += 1
    end
    return x
end


# It performs the product EDE^Tv.
function custom_product(edges_list, E, D, b)
    x = zeros(size(E, 2))
    for (i, (src, dst)) in enumerate(edges_list)
        x[i] = D[i] * (b[src] - b[dst])
    end
    return E * x
end


function conjugate_gradient(edges_list, E, D, b, tol, max_iter)
    n = size(b, 1)
    x = zeros(n)
    d = b
    r_old = b
    D_inv = ones(size(D, 1)) ./ D
    tol *= norm(b)

    errors = []

    for _ in 1:max_iter
        rr_old = r_old'r_old

        if sqrt(rr_old) <= tol
            print("fast")
            return x, errors
        end
        
        Ld = custom_product(edges_list, E, D_inv, d)

        α = rr_old / (d'Ld)
        x = x + α * d
        r = r_old - α * Ld
        β = (r'r) / rr_old
        d = r + β * d

        r_old = r
        append!(errors, norm(r))
    end

    return x, errors
end
