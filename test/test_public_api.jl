using Test
using PhaseFields
using OpenCALPHAD   # already in test/Project.toml; needed for the clash test below

# PhaseFields and OpenCALPHAD each export these names, and in each case the two
# are separate generic functions rather than one shared generic. Their methods
# dispatch on disjoint argument types, so the clash is in the name only: Julia
# refuses to merge two distinct generics, and a qualified call resolves fine.
# Both exports are legitimate inside their own package, so neither is removed;
# tidying the exports is breaking and waits for v0.3.0 (see GAP-21, which also
# covers the L1/L2 non-exported, L3 exported asymmetry of the plotting API).
# This set is frozen so a new clash cannot appear unnoticed.
const KNOWN_CLASHES = Set([:chemical_potential, :savefig_publication])

# Exported functions that intentionally carry no method until their weak
# dependency is loaded. Each maps to the extension that supplies the methods.
const WEAKDEP_STUBS = Dict(
    :GridapDomain                => :GridapExt,
    :calphad_driving_force       => :OpenCALPHADExt,
    :calphad_chemical_potential  => :OpenCALPHADExt,
    :calphad_diffusion_potential => :OpenCALPHADExt,
    :calphad_free_energy         => :OpenCALPHADExt,
    :create_calphad_allen_cahn   => :OpenCALPHADExt,
    :create_calphad_kks_model    => :OpenCALPHADExt,
    :create_calphad_wbm_model    => :OpenCALPHADExt,
)

# Exported functions with no method anywhere and no extension declared to supply
# one (#29). src/plotting.jl points at ext/PhaseFieldsPlotsExt.jl, which does not
# exist and is absent from Project.toml [weakdeps] and [extensions]. Calling
# either one always raises MethodError. Dropping the exports is breaking, so it
# waits for v0.3.0; until then this set freezes the debt so no new orphan slips in.
const KNOWN_ORPHANS = Set([:plot_field, :animate_field])

exported_functions() =
    [s for s in names(PhaseFields)
     if s !== :PhaseFields && getproperty(PhaseFields, s) isa Function]

methodless_exports() =
    Set(s for s in exported_functions()
        if isempty(methods(getproperty(PhaseFields, s))))

ext_loaded(name) = Base.get_extension(PhaseFields, name) !== nothing

@testset "public API surface" begin
    methodless = methodless_exports()

    @testset "every method-less export is accounted for" begin
        for s in methodless
            accounted = s in KNOWN_ORPHANS ||
                (haskey(WEAKDEP_STUBS, s) && !ext_loaded(WEAKDEP_STUBS[s]))
            @test accounted
        end
    end

    @testset "a loaded extension populates its stubs" begin
        for (sym, ext) in WEAKDEP_STUBS
            ext_loaded(ext) || continue
            @test !isempty(methods(getproperty(PhaseFields, sym)))
        end
    end

    @testset "the orphan set is frozen" begin
        # Every known orphan is still method-less. If one gains a method, delete
        # it from KNOWN_ORPHANS.
        @test KNOWN_ORPHANS ⊆ methodless
        # Nothing else is method-less except unloaded weak-dep stubs. A new
        # orphaned export fails here.
        @test setdiff(methodless, KNOWN_ORPHANS) ⊆ Set(keys(WEAKDEP_STUBS))
    end

    @testset "orphans are debt, not design" begin
        # Flips to an Unexpected Pass once v0.3.0 resolves plot_field/animate_field.
        # When that happens, delete this testset together with KNOWN_ORPHANS.
        @test_broken isempty(KNOWN_ORPHANS)
    end

    # GAP-21
    @testset "the export clash set with OpenCALPHAD is frozen" begin
        pf = Set(s for s in names(PhaseFields) if s !== :PhaseFields)
        oc = Set(s for s in names(OpenCALPHAD) if s !== :OpenCALPHAD)
        clashes = intersect(pf, oc)

        # A new clash means users must qualify one more name. Document it in
        # docs/src/integration/calphad.md and add it here, or rename one side.
        @test clashes == KNOWN_CLASHES

        # Each clash really is two different functions, not one shared generic.
        for s in KNOWN_CLASHES
            @test getproperty(PhaseFields, s) !== getproperty(OpenCALPHAD, s)
        end
    end
end
