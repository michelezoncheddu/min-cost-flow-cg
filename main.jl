using Distributions
using IterativeSolvers
using LinearAlgebra  # For identity matrix.
using Plots
using SparseArrays


include("InputGenerator.jl")
include("ConjugateGradient.jl")


# It performs the product EDE^Tv.
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


# Radius of the range.
function test(dir, filename, distr_type=0, rad=5, max_iter=10000, tol=1e-5)
    # Read DIMACS input.
    edges_list, b, E, D = readDIMACS("$dir/$filename.dmx")
    n_edges = length(edges_list)

    if distr_type == 1
        distr = Uniform(0, exp(rad))
        D = rand(distr, n_edges)
    elseif distr_type == 2
        distr = Uniform(-rad, rad)
        D = exp.(rand(distr, n_edges))
    end

    # Compute Laplacian matrix.
    L = E * sparse(1:n_edges, 1:n_edges, inv.(D)) * E'

    #eigs = eigvals(Array(L))
    #println(sqrt(eigs[end] / eigs[2]))  # Condition number


    # Run our CG
    time = @elapsed begin
        x, xs, residuals, status = conjugate_gradient(custom_product, edges_list, E, D, b, tol, max_iter)
    end
    if cmp(status, "Error") == 0
        return Nothing
    end
    println("Our CG: $status in $(length(residuals)) steps, took $time seconds")


    # Run Julia CG
    time = @elapsed begin
        x, residuals = cg(L, b, reltol=tol, maxiter=max_iter, log=true)
    end
    println("Julia CG: took $(residuals.iters) steps, took $time seconds")


    # Run Julia GMRES
    time = @elapsed begin
        x, residuals = gmres(L, b, reltol=tol, maxiter=max_iter, log=true)
    end
    println("Julia GMRES: took $(residuals.iters) steps, took $time seconds")


    # Plots
    #eigs_plot = plot(2:length(eigs), log10.(abs.(eigs[2:end])), legend=false, xlabel="i-th eigenvalue", ylabel="log(λᵢ)")
    #residuals_plot = plot(1:length(residuals), log10.(residuals), legend=false)
    #savefig(residuals_plot, "$dir/plots/$filename.png")
end


tool = "netgen"
dir = "datasets/$tool"


test(dir, "netgen100-10", 2, 15)
