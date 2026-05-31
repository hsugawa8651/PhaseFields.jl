using Documenter
using PhaseFields
using PythonPlot                   # loads PhaseFieldsPythonPlotExt so its docstrings render
PythonPlot.matplotlib.use("Agg")  # headless backend for CI

# The PythonPlot extension carries the docstrings for plot_on_axis! /
# figure_publication / savefig_publication; include it so @docs can render them.
const PFPythonPlotExt = Base.get_extension(PhaseFields, :PhaseFieldsPythonPlotExt)

makedocs(
    sitename = "PhaseFields.jl",
    authors = "Hiroharu Sugawara <hsugawa@tmu.ac.jp>",
    repo = Documenter.Remotes.GitHub("hsugawa8651", "PhaseFields.jl"),
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
    ),
    modules = [PhaseFields, PFPythonPlotExt],
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
        "Plotting" => "plotting.md",
        "Reference" => "reference/common.md",
    ],
)

deploydocs(
    repo = "github.com/hsugawa8651/PhaseFields.jl.git",
    devbranch = "main",
)
