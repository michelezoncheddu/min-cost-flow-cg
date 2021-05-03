using LinearAlgebra  # For identity matrix.
using SparseArrays

using LinearAlgebra
using .ConjugateGradient
using .InputGenerator

#read DIMACS input
n_nodes, n_edges, edges_list, b, E, D = readDIMACS("netgen-1000-1-1-a-a-ns.dmx")



# compute Laplacian
L = E * inv(Array(sparse(Array{Int64}(1:n_edges) ,Array{Int64}(1:n_edges), D))) * E'

if rank(L) != (n_nodes -1)
    print("Rank must be n - 1")
end



x = conjugate_gradient(edges_list, E, D, b)

L * x ≈ b

norm((L * x - b)) / norm(b)