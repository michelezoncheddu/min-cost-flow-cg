using LinearAlgebra  # For identity matrix.
using SparseArrays
using Distributions

include("InputGenerator.jl")
include("ConjugateGradient.jl")


# Read DIMACS input.
n_nodes, n_edges, edges_list, b, E, D = readDIMACS("datasets/1000/netgen-1000-3-2-a-a-s.dmx")

if rank(E) != (n_nodes-1)
    print("ERROR: The graph must be connected.")
end

rad = 50
distr = Uniform(-rad, rad)
D = exp.(rand(distr, n_edges))

# Compute Laplacian matrix.
L = E * inv(Array(sparse(Array{Int64}(1:n_edges), Array{Int64}(1:n_edges), D))) * E'

x = conjugate_gradient(edges_list, E, D, b)

L * x ≈ b

norm((L * x - b)) / norm(b)

eigs = eigvals(L)
eigs[end] / eigs[2]
