using Distributions
using IterativeSolvers
using LinearAlgebra  # For identity matrix.
using Plots
using SparseArrays


include("InputGenerator.jl")
include("ConjugateGradient.jl")


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
Tests different linear system solvers for a quadratic MCF problem
read in a DIMACS file.

# Input parameters
    - dir: directory in which search the input file;
    
    - filename: name of the input file. The file must be DIMACS compliant;
    
    - [distr_type = 0]: 1 = uniform distribution: (0, e^range),
                        2 = exponential distribution: e^(-range, range),
                        native distribution otherwise;
    
    - [range = 5]: range of the interval for the distributions;
    
    - [max_iter = 10000]: maximum number of steps for iterative algorithms;
    
    - [tol = 1e-8]: accuracy for the stopping criterion for iterative algorithms.
"""
function test(dir, filename; distr_type = 0, range = 5, max_iter = 10000, tol = 1e-8)
    # Read DIMACS input.
    edges_list, b, E, D = readDIMACS("$dir/$filename.dmx")
    n_edges = length(edges_list)

    if distr_type == 1  # Uniform
        distr = Uniform(0, exp(range))
        D = rand(distr, n_edges)
    elseif distr_type == 2  # Exponential
        distr = Uniform(-range, range)
        D = exp.(rand(distr, n_edges))
    end

    # Compute Laplacian matrix.
    L = E * sparse(1:n_edges, 1:n_edges, inv.(D)) * E'

    #eigs = eigvals(Array(L))
    #println(sqrt(eigs[end] / eigs[2]))  # Condition number


    # Run our CG
    time = @elapsed begin
        x, residuals, status = conjugate_gradient(custom_product, edges_list, E, D, b, tol, max_iter, verbose = false)
    end
    if cmp(status, "error") == 0
        return
    end
    println("Our CG: $status in $(length(residuals)) steps, took $time seconds, $(time/length(residuals)*1000) ms")


    # DEBUG ------------------------------------------------------------------------------------------------
    return


    # Run Julia CG
    time = @elapsed begin
        x, residuals = cg(L, b, reltol = tol, maxiter = max_iter, log = true)
    end
    println("Julia CG: took $(residuals.iters) steps, took $time seconds")


    # Run Julia GMRES
    time = @elapsed begin
        x, residuals = gmres(L, b, reltol = tol, maxiter = max_iter, log = true)
    end
    println("Julia GMRES: took $(residuals.iters) steps, took $time seconds")


    # Run sparse LU
    time = @elapsed begin
        xLU = lu(L) \ b
    end
    println("LU: took $time seconds")


    # Run sparse LDL
    time = @elapsed begin
        xLDL = ldlt(L) \ b
    end
    println("LDL: took $time seconds")


    # Plots
    #eigs_plot = plot(2:length(eigs), log10.(abs.(eigs[2:end])), legend=false, xlabel="i-th eigenvalue", ylabel="log(λᵢ)")
    #residuals_plot = plot(1:length(residuals), log10.(residuals), legend=false)
    #savefig(residuals_plot, "$dir/plots/$filename.png")
end


tool = "netgen"
dir = "datasets/$tool"

# Example test
test(dir, "netgen100-10", distr_type = 2, range = 10, max_iter = 100000)
