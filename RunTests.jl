include("main.jl")


dir = "test_datasets"

# Test parameters -------------------------
distr_type = 2
range = 10
max_iter = 10000
# -----------------------------------------

for file in readdir(dir)
    if !endswith(file, ".dmx")
        continue
    end
    
    println("\nTesting $file")
    test("$dir/$file", distr_type = distr_type, range = range, max_iter = max_iter)
end
