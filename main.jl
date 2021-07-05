using LinearAlgebra  # For identity matrix.
using SparseArrays
using Distributions
using Plots


include("InputGenerator.jl")
include("ConjugateGradient.jl")


# Read DIMACS input.
n_nodes, n_edges, edges_list, b, E, D = readDIMACS("datasets/road_flow_02_DE_a.min")

if rank(E) != (n_nodes-1)
    print("ERROR: The graph must be connected.")
end

#=rad = 50
distr = Uniform(-rad, rad)
D = exp.(rand(distr, n_edges))=#

# Compute Laplacian matrix.
L = Array(E * sparse(Array{Int64}(1:n_edges), Array{Int64}(1:n_edges), inv.(D)) * E')

eigs = eigvals(L)
max_iter = n_nodes #sqrt(eigs[end] / eigs[2])

x, errors = conjugate_gradient(edges_list, E, D, b, 1e-5, max_iter)

L * x ≈ b

print(norm((L * x - b)) / norm(b))

p = plot(1:length(eigs), log10.(abs.(eigs[1:end])), legend=false, xlabel="i-th eigenvalue", ylabel="log(λᵢ)")
p = plot(1:length(errors), log10.(errors), legend=false)

#savefig(p, "plot.png")
