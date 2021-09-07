using Printf
using SparseArrays


"""
Performs the product EDE^T * v, exploiting the structure
of the incidence matrix.

# Input parameters
    - edges_list: list of pairs (src => dst);
    
    - E: incidence matrix;

    - D: weight/cost list;

    - v: vector.

# Returns
    - The vector y = EDE^T * v.
"""
function custom_product(edges_list::Array{<:Pair{<:Integer, <:Integer}, 1},
                        E::SparseMatrixCSC{<:Real, <:Integer},
                        D::Array{<:Real, 1},
                        v::Array{<:Real, 1})
    x = zeros(size(E, 2))
    for (i, (src, dst)) in enumerate(edges_list)
        x[i] = D[i] * (v[src] - v[dst])
    end
    return E * x
end


"""
Implements a conjugate gradient algorithm for solving
the linear system EDE^Tx = b.

# Input parameters
    - product_function: callback that performs EDE^T * d;

    - edges_list: list of pairs (src => dst);

    - E: incidence matrix;

    - D: vector of weights/costs of the edges;

    - b: rhs of the linear system;

    - tol: accuracy in the stopping criterion. It must be >= 0;

    - max_iter: maximum number of iterations, regardless of the accuracy.
                it must be > 0;

    - [verbose = false]: print method information for each iteration.

# Returns
    x::Array, residuals::Array, status::String

    - x: Nothing if max_iter <= 0 or tol < 0,
         approximated solution of the linear otherwise;
    
    - residuals: Nothing if max_iter <= 0 or tol < 0,
                 history of residuals otherwise;
    
    - status: "error" if max_iter <= 0 or tol < 0,
              "converged" if the accuracy is reached within max_iter steps,
              "stopped" otherwise.
"""
function conjugate_gradient(product_function,
                            edges_list::Array{<:Pair{<:Integer, <:Integer}, 1},
                            E::SparseMatrixCSC{<:Real, <:Integer},
                            D::Array{<:Real, 1},
                            b::Array{<:Real, 1},
                            tol::Real,
                            max_iter::Integer;
                            verbose::Bool = false)
    if max_iter <= 0
        println("ERROR: max_iter should be > 0")
        return Nothing, Nothing, "error"
    end

    if tol < 0
        println("ERROR: tol should be ≥ 0")
        return Nothing, Nothing, "error"
    end

    n = size(b, 1)
    x = zeros(n)   # Approximated solution
    r_old = b      # Residual of the previous iteration
    d = b          # Direction of the iteration
    rr_old = r_old'r_old

    D_inv = ones(size(D, 1)) ./ D  # D^(-1)
    tol *= norm(b)
    residuals = []  # History of residuals

    for iteration in 1:max_iter
        rr_old = r_old'r_old  # Scalar product of the previous residual

        # ||r_i|| <= tol * ||r_0||
        if sqrt(rr_old) <= tol
            return x, residuals, "converged"
        end
        
        # Ld = EDE^Td
        Ld = product_function(edges_list, E, D_inv, d)

        α = rr_old / (d'Ld)  # Step size
        x .+= α .* d
        r = r_old .- α .* Ld
        β = (r'r) / rr_old
        d = r .+ β .* d

        r_old = r
        push!(residuals, norm(r))
        verbose && @printf("%3d\t%.2e\t%.2e\t%.2f\n", iteration, norm(r), α, β)
    end

    if sqrt(rr_old) <= tol
        status = "converged"
    else
        status = "stopped"
    end
    return x, residuals, status
end
