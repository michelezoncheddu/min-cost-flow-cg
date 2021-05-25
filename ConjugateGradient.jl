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


function conjugate_gradient(edges_list, E, D, b::Vector)
    n = size(b, 1)
    x = zeros(n)
    d = b
    r_old = b
    D_inv = ones(size(D, 1)) ./ D

    for i in 1:n
        rr_old = r_old'r_old
        Ld = E * first_part_product(edges_list, E, D_inv, d)

        alpha = rr_old / (d'Ld)
        x = x + alpha * d
        r = r_old - alpha * Ld
        beta = (r'r) / rr_old
        d = r + beta * d

        r_old = r
    end

    return x
end
