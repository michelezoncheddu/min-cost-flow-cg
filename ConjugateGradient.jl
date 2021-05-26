using SparseArrays


function first_part_product(edges_list, E, D, b)
    x = zeros(size(E, 2))
    i = 1
    for (src, dst) in edges_list
        x[i] += D[i] * b[src]
        x[i] -= D[i] * b[dst]
        i = i + 1
    end
    return x
end


function conjugate_gradient(edges_list, E, D, b, tol)
    n = size(b, 1)
    x = zeros(n)
    d = b
    r_old = b
    D_inv = ones(size(D, 1)) ./ D

    errors = []

    for i in 1:size(E, 2)  # TODO: check dimension
        rr_old = r_old'r_old

        if sqrt(rr_old) <= tol  # TODO: check
            return x, errors
        end
        
        Ld = E * first_part_product(edges_list, E, D_inv, d)

        alpha = rr_old / (d'Ld)
        x = x + alpha * d
        r = r_old - alpha * Ld
        beta = (r'r) / rr_old
        d = r + beta * d

        r_old = r
        append!(errors, norm(r))
    end

    return x, errors
end
