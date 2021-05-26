using LinearAlgebra  # For identity matrix.
using SparseArrays
using Distributions
using Plots


include("InputGenerator.jl")
include("ConjugateGradient.jl")


# Read DIMACS input.
n_nodes, n_edges, edges_list, b, E, D = readDIMACS("datasets/3000/netgen-3000-1-1-a-b-ns.dmx")

if rank(E) != (n_nodes-1)
    print("ERROR: The graph must be connected.")
end

rad = 50
distr = Uniform(-rad, rad)
D = exp.(rand(distr, n_edges))

# Compute Laplacian matrix.
L = E * inv(Array(sparse(Array{Int64}(1:n_edges), Array{Int64}(1:n_edges), D))) * E'

x, errors = conjugate_gradient(edges_list, E, D, b, 1e-5)

L * x ≈ b

norm((L * x - b)) / norm(b)

eigs = eigvals(L)
print(eigs[end] / eigs[2])

#plot(1:length(eigs), log10.(abs.(eigs[1:end])), legend=false)
#plot(1:length(errors), log10.(errors), legend=false)
