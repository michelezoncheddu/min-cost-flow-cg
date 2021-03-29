using LightGraphs
using LinearAlgebra  # For identity matrix.
using SparseArrays

using .ConjugateGradient
using .InputGenerator

n_nodes = 3
n_edges = 4

G = create_graph(n_nodes, n_edges)
E = incidence_matrix(G)
D = SparseMatrixCSC(1.0I, n_edges, n_edges)

L = E * inv(Matrix(D)) * E'

b = vector_from_image(L)

x = conjugate_gradient(L, b)

L * x ≈ b
