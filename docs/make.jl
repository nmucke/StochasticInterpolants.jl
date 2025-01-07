using StochasticInterpolants
using Documenter

DocMeta.setdocmeta!(StochasticInterpolants, :DocTestSetup, :(using StochasticInterpolants); recursive=true)

makedocs(;
    modules=[StochasticInterpolants],
    authors="ntm <nmucke@gmail.com> and contributors",
    sitename="StochasticInterpolants.jl",
    format=Documenter.HTML(;
        canonical="https://nmucke.github.io/StochasticInterpolants.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/nmucke/StochasticInterpolants.jl",
    devbranch="main",
)
