using LinearAlgebra  # For identity matrix.
using SparseArrays

using .ConjugateGradient
using .InputGenerator

#read DIMACS input
n_nodes, n_edges, b, E, D = readDIMACS("netgen-1000-1-1-a-a-ns.dmx")

# compute Laplacian
L = E * inv(Array(D)) * E'

if rank(L) != (n_nodes -1)
    print("Rank must be n - 1")
end

x = conjugate_gradient(L, b)

L * x ≈ b

norm((L * x - b)) / norm(b)