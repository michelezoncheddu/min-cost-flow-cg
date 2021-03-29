module InputGenerator

    using LightGraphs

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

    export create_graph, vector_from_image
end
