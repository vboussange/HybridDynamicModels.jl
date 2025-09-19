using Documenter, Lux, Turing, ComponentArrays, HybridDynamicModels

readme_str = read(joinpath(@__DIR__, "..", "README.md"), String)
readme_str = replace(readme_str, "> [!CAUTION]\n> " => "!!! warning\n    ")
write(joinpath(@__DIR__, "src", "index.md"), readme_str)

mathengine = Documenter.MathJax()
DocMeta.setdocmeta!(
    HybridDynamicModels, :DocTestSetup, :(using HybridDynamicModels); recursive=true)

makedocs(;
    modules=[HybridDynamicModels],
    authors="Victor Boussange",
    sitename="HybridDynamicModels.jl",
    linkcheck=false,
    clean=true,
    format=Documenter.HTML(;
        assets=["assets/favicon.ico"],
    ),
    pages=[
        "Home" => "index.md",
        "api.md",
        "dev_guide.md",
    ],
    doctest = false,
)

deploydocs(;
    repo="github.com/vboussange/HybridDynamicModels.jl",
    devbranch="main",
    push_preview=true
)