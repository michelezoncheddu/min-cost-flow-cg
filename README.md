## Folders
The datasets folder is structured this way:
- netgen
    - edges
    - nodes
        - density
        - avgdeg
- gridgen
    - edges
    - nodes
        - density
        - avgdeg
- roads

The name format of any test file within the netgen (or gridgen) folder is: netgenX-Y.dmx (gridgenX-Y.dmx), where X is X is the number of edges in the edges folder, and the number of nodes in the nodes folder, and Y is the density in the density folder, and the average degree in the average degree folder.


## Files
**ConjugateGradient.jl**: implements the CG and the specialized product;

**main.jl**: executes CG and other methods;

**InputParser.jl**: reads the DIMACS files and builds the graph data structures;

**RunTests.jl**: runs the test from a specified folder (default is test_datasets).


## Required packages 
* Distributions ≥ v0.25.1
* IterativeSolvers ≥ v0.9.1

To add a package, run in a Julia environment the following commands:
```
> using Pkg
> Pkg.add("some_package")
```


## How to run the project
```
> julia RunTests.jl
```

This will run the tests using all the files in the test_datasets folder. You can put any DIMACS file inside it.
