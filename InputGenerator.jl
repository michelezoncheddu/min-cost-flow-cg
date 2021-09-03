using SparseArrays


"""
Parses a DIMACS file.

# Input parameters
    - filename: name of the input file. The file must be DIMACS compliant.

# Returns
    edges_list, b, E, D

    - edges_list: list of pairs (src => dst);

    - b: vector of supply/demand balances of the nodes;

    - E: incidence matrix of the graph;

    - D: vector of weights/costs of the edges.
"""
function readDIMACS(filename)
    edges_counter = 1  # For edge labelling
    b = Float64[]
    
    # For building E
    e_x = Int64[]
    e_y = Int64[]
    e_v = Float64[]

    edges_list = Pair{Int64, Int64}[]

    D = Float64[]

    open(filename, "r") do f
        for line in eachline(f)
            tokens = split(line, " ")
            if cmp(convert(String, tokens[1]), "p") == 0  # Problem
                n_nodes = parse(Int64, tokens[3])
                b = zeros(n_nodes)
            elseif cmp(convert(String, tokens[1]), "n") == 0  # Node
                id = parse(Int64, tokens[2])
                b[id] = parse(Float64, tokens[3])
            elseif cmp(convert(String, tokens[1]), "a") == 0  # Edge
                src = parse(Int64, tokens[2])
                dst = parse(Int64, tokens[3])
                push!(edges_list, src => dst)
                push!(e_x, src)
                push!(e_y, edges_counter)
                push!(e_v, 1)
                push!(e_x, dst)
                push!(e_y, edges_counter)
                push!(e_v, -1)

                weight = parse(Float64, tokens[6])
  
                push!(D, max(weight, 1))
                edges_counter += 1
            end
        end
    end
    
    E = sparse(e_x, e_y, e_v)
    return edges_list, b, E, D
end
