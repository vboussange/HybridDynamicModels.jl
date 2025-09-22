using Documenter, Lux, Turing, ComponentArrays, HybridDynamicModels
using Random
using Weave

example_scripts = ["data_loading.jmd",
    "sgd_example.jmd", 
    "mcsampling_example.jmd",
    "customtraining_example.jmd"]
out_path = joinpath(@__DIR__, "..", "docs", "src", "examples")
isdir(out_path) || mkpath(out_path)
println("Weaving example scripts...")
for script in example_scripts
    weave(joinpath(@__DIR__, "..", "examples", script);
        out_path,
        doctype = "github")
end

weaved_examples = [joinpath("examples", replace(script, ".jmd" => ".md"))
                   for script in example_scripts]

readme_str = read(joinpath(@__DIR__, "..", "README.md"), String)
readme_str = replace(readme_str, "> [!CAUTION]\n> " => "!!! warning\n    ")
write(joinpath(@__DIR__, "src", "index.md"), readme_str)

mathengine = Documenter.MathJax()
DocMeta.setdocmeta!(
    HybridDynamicModels, :DocTestSetup, :(using HybridDynamicModels); recursive = true)

makedocs(;
    modules = [HybridDynamicModels],
    authors = "Victor Boussange",
    sitename = "HybridDynamicModels.jl",
    linkcheck = false,
    clean = true,
    format = Documenter.HTML(;
        assets = ["assets/favicon.ico"],
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages = [
        "Home" => "index.md",
        "Tutorials" => weaved_examples,
        "api.md",
    ],
    doctest = false
)

deploydocs(;
    repo = "github.com/vboussange/HybridDynamicModels.jl",
    devbranch = "main",
    push_preview = true
)