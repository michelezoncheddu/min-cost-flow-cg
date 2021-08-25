using Plots
using LinearAlgebra  # For identity matrix.
using SparseArrays
using Distributions
using Plots


include("InputGenerator.jl")
include("ConjugateGradient.jl")


# It performs the product EDE^Tv.
function custom_product(edges_list, E, D, b)
    x = zeros(size(E, 2))
    for (i, (src, dst)) in enumerate(edges_list)
        x[i] = D[i] * (b[src] - b[dst])
    end
    return E * x
end


function test(dir, filename)
    # Read DIMACS input.
    n_nodes, n_edges, edges_list, b, E, D = readDIMACS(dir * filename * ".dmx")

    if rank(E) != (n_nodes-1)
        println("ERROR: The graph must be connected.")
    end

    rad = 15
    distr_exp = Uniform(-rad, rad)
    distr_uni = Uniform(0, exp(rad))
    #D = exp.(rand(distr_exp, n_edges))
    D = rand(distr_uni, n_edges)

    # Compute Laplacian matrix.
    L = E * sparse(1:n_edges, 1:n_edges, inv.(D)) * E'

    #eigs = eigvals(Array(L))
    #println(sqrt(eigs[end] / eigs[2]))  # Condition number
    max_iter = 10000 #sqrt(eigs[end] / eigs[2])

    time = @elapsed begin
        x, residuals = conjugate_gradient(custom_product, edges_list, E, D, b, 1e-8, max_iter)
    end
    println(time)  # Seconds
    println(time / length(residuals) * 1000)  # Milliseconds

    #println(norm((L * x - b)) / norm(b))

    #p1 = plot(2:length(eigs), log10.(abs.(eigs[2:end])), legend=false, xlabel="i-th eigenvalue", ylabel="log(λᵢ)")
    p = plot(1:length(residuals), log10.(residuals), legend=false)

    savefig(p, dir * "plots/" * filename * ".png")

    return length(residuals)
end


tool = "netgen/"
dir = "datasets/" * tool
files = readdir(dir)


i = 0
f = open("results.txt", "w")
for file in files
    if !endswith(file, ".dmx")
        continue
    end
    
    #break

    println("Testing " * file)
    filename = chop(file, tail=4)  # Remove .dmx extension
    steps = test(dir, filename)

    write(f, string(steps))

    i += 1
    if i % 4 == 0
        write(f, "\n")
    else
        write(f, " ")
    end
end
close(f)

#test(dir, "netgen200-70")
