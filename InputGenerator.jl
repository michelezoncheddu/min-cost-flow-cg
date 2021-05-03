module InputGenerator

    using LightGraphs
    using SparseArrays

    function create_graph(n_nodes::Int, n_edges::Int)
        graph = SimpleDiGraph(n_nodes, n_edges)

        while !is_strongly_connected(graph)
            graph = SimpleDiGraph(n_nodes, n_edges)
        end

        return graph
    end


    function vector_from_image(matrix::Matrix)
        return 2 * matrix[:, 1] + 5 * matrix[:, 2]
    end

    function readDIMACS(file_name)
        n_nodes = n_edges = 0
        edges_counter = 1 # edge labelling
        b = Float64[] # target vector
        
        # incidence matrix E
        e_x = Int64[] 
        e_y = Int64[]
        e_v = Float64[]
        
        edges_list = Pair{Int64, Int64}[]   # list of edges

        diag = Float64[] # weight D
        
        open(file_name, "r") do f
            for line in eachline(f)
                tokens = split(line, " ")
                if cmp(convert(String,tokens[1]), "p")==0 # node edges
                    n_nodes = parse(Int64,tokens[3])
                    n_edges = parse(Int64,tokens[4])
                    b = zeros(n_nodes)
                end
                if cmp(convert(String,tokens[1]), "n")==0 # target vector
                    id = parse(Int64,tokens[2])
                    b[id] = parse(Int64,tokens[3])
                end
                if cmp(convert(String,tokens[1]), "a")==0 # edges
                    # for building E
                    src = parse(Int64,tokens[2])
                    dst = parse(Int64,tokens[3])
                    push!(edges_list, src => dst)
                    append!(e_x, src)
                    append!(e_y, edges_counter)
                    append!(e_v, 1)
                    append!(e_x, dst)
                    append!(e_y, edges_counter)
                    append!(e_v, -1)
                    # for building D
                    weight = parse(Float64,tokens[6])
                    append!(diag, weight)
                    edges_counter = edges_counter + 1
                end
            end
        end
       
        E = sparse(e_x, e_y, e_v)
        return n_nodes, n_edges, edges_list, b, E, diag
    end
    

    export create_graph, vector_from_image, readDIMACS
end
