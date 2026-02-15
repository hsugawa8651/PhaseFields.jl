using Documenter
using PhaseFields

makedocs(
    sitename = "PhaseFields.jl",
    authors = "Hiroharu Sugawara <hsugawa@tmu.ac.jp>",
    repo = Documenter.Remotes.GitHub("hsugawa8651", "PhaseFields.jl"),
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
    ),
    modules = [PhaseFields],
    checkdocs = :exports,  # Only check exported symbols
    warnonly = [:missing_docs, :cross_references],  # Don't fail on missing docs or unit brackets
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Models" => [
            "Allen-Cahn" => "models/allen_cahn.md",
            "Cahn-Hilliard" => "models/cahn_hilliard.md",
            "KKS" => "models/kks.md",
            "WBM" => "models/wbm.md",
            "Thermal" => "models/thermal.md",
        ],
        "Integration" => [
            "DifferentialEquations.jl" => "integration/diffeq.md",
            "CALPHAD Coupling" => "integration/calphad.md",
        ],
        "Reference" => "reference/common.md",
    ],
)

deploydocs(
    repo = "github.com/hsugawa8651/PhaseFields.jl.git",
    devbranch = "main",
)
