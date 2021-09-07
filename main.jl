using Distributions
using IterativeSolvers
using LinearAlgebra
#using Plots
using SparseArrays


include("InputParser.jl")
include("ConjugateGradient.jl")


"""
Tests different linear system solvers for a quadratic MCF problem
read in a DIMACS file.

# Input parameters
    - dir: directory in which to search the input file;
    
    - filename: name of the input file. The file must be DIMACS compliant;
    
    - [distr_type = 0]: 1 = uniform distribution: (0, e^range),
                        2 = exponential distribution: e^(-range, range),
                        native distribution otherwise;
    
    - [range = 5]: range of the interval for the distributions;
    
    - [max_iter = 10000]: maximum number of steps for iterative algorithms;
    
    - [tol = 1e-8]: accuracy for the stopping criterion for iterative algorithms;

    - [direct = false]: runs also the direct methods for solving the linear system.
"""
function test(filename; distr_type = 0, range = 5, max_iter = 10000, tol = 1e-8, direct = false)
    filename = chop(filename, tail = 4)  # Remove .dmx extension.
    edges_list, b, E, D = readDIMACS("$filename.dmx")  # Read DIMACS input.
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
    println("\tOur CG: $status in $(length(residuals)) steps, took $time seconds")


    # Run Julia CG
    time = @elapsed begin
        xCG, residuals = cg(L, b, reltol = tol, maxiter = max_iter, log = true)
    end
    println("\tJulia CG: took $(residuals.iters) steps, took $time seconds")


    # Run Julia GMRES
    time = @elapsed begin
        xGMRES, residuals = gmres(L, b, reltol = tol, maxiter = max_iter, log = true)
    end
    println("\tJulia GMRES: took $(residuals.iters) steps, took $time seconds")


    if !direct
        return
    end

    # Run sparse LU
    time = @elapsed begin
        xLU = lu(L) \ b
    end
    println("\tLU: took $time seconds")


    # Run sparse LDL
    time = @elapsed begin
        xLDL = ldlt(L) \ b
    end
    println("\tLDL: took $time seconds")

    
    # Plots
    #eigs_plot = plot(2:length(eigs), log10.(abs.(eigs[2:end])), legend=false, xlabel="i-th eigenvalue", ylabel="log(λᵢ)")
    #residuals_plot = plot(1:length(residuals), log10.(residuals), legend=false)
    #savefig(residuals_plot, "$filename.png")
end
