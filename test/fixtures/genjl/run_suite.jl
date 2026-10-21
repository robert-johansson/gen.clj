# Run Gen.jl 0.4.8's own test suite once, from the installed copy, through a
# recording test set, and write each testset's pass/fail/error counts as EDN.
#
#   env JULIA_PKG_OFFLINE=true JULIA_NUM_THREADS=1 JULIA_DEPOT_PATH=<scratch depot>:~/.julia \
#     nice -n 15 julia --project=<m5-julia> run_suite.jl <Gen's test dir> <out dir>
#
# runtests.jl, then inference/enumerative.jl (which runtests.jl never includes).
using Test

const TESTDIR = abspath(ARGS[1])
const OUTDIR = abspath(ARGS[2])

mutable struct RecTS <: Test.AbstractTestSet
    description::String
    results::Vector{Any}
    children::Vector{Any}
    t0::Float64
    t1::Float64
end
RecTS(desc::AbstractString; kwargs...) = RecTS(String(desc), Any[], Any[], time(), 0.0)

function short(s; n=300)
    s = replace(string(s), "\n" => " ")
    length(s) > n ? first(s, n) * "…" : s
end

function Test.record(ts::RecTS, t::Test.Result)
    push!(ts.results, t)
    if t isa Test.Fail || t isa Test.Error
        src = t.source
        println(stderr, "  ", typeof(t).name.name, " in \"", ts.description, "\" at ",
                src === nothing ? "?" : string(src.file, ":", src.line), ": ", short(t isa Test.Fail ? t.orig_expr : t.value; n=200))
    end
    t
end
Test.record(ts::RecTS, child::RecTS) = (push!(ts.children, child); child)

function Test.finish(ts::RecTS)
    ts.t1 = time()
    if Test.get_testset_depth() > 0
        Test.record(Test.get_testset(), ts)
    end
    np = count(r -> r isa Test.Pass, ts.results)
    nf = count(r -> r isa Test.Fail, ts.results)
    ne = count(r -> r isa Test.Error, ts.results)
    println(stderr, "done \"", ts.description, "\": pass ", np, " fail ", nf, " error ", ne,
            " (", round(ts.t1 - ts.t0; digits=1), " s)")
    flush(stderr)
    ts
end

# --- EDN ---------------------------------------------------------------------
include("edn.jl")

relfile(f) = (s = string(f); startswith(s, TESTDIR) ? s[length(TESTDIR)+2:end] : s)

function rows(ts::RecTS, path::Vector{String}=String[])
    p = vcat(path, ts.description)
    srcs = [r.source for r in ts.results if hasproperty(r, :source) && r.source isa LineNumberNode]
    files = unique([relfile(s.file) for s in srcs])
    lines = [s.line for s in srcs]
    bad = [Dict(:kind => (r isa Test.Fail ? :fail : :error),
                :type => r.test_type,
                :line => (r.source === nothing ? nothing : r.source.line),
                :file => (r.source === nothing ? nothing : relfile(r.source.file)),
                :expr => short(r.orig_expr),
                :value => short(r isa Test.Fail ? r.value : r.value))
           for r in ts.results if r isa Test.Fail || r isa Test.Error]
    own = Dict(:path => p, :files => files,
               :lines => isempty(lines) ? nothing : [minimum(lines), maximum(lines)],
               :pass => count(r -> r isa Test.Pass, ts.results),
               :fail => count(r -> r isa Test.Fail, ts.results),
               :error => count(r -> r isa Test.Error, ts.results),
               :broken => count(r -> r isa Test.Broken, ts.results),
               :seconds => round(ts.t1 - ts.t0; digits=2),
               :problems => bad)
    out = Any[own]
    for c in ts.children
        append!(out, rows(c, p))
    end
    out
end

# --- the run -----------------------------------------------------------------
cd(TESTDIR)
top = @testset RecTS "Gen.jl 0.4.8" begin
    @testset RecTS "runtests.jl" begin
        include(joinpath(TESTDIR, "runtests.jl"))
    end
    @testset RecTS "inference/enumerative.jl (not included by runtests.jl)" begin
        include(joinpath(TESTDIR, "inference", "enumerative.jl"))
    end
end

rs = rows(top)
open(joinpath(OUTDIR, "upstream-run.edn"), "w") do io
    println(io, ";; Gen.jl 0.4.8's test suite, run once from ", TESTDIR)
    println(io, ";; Julia ", VERSION, ", ", Sys.MACHINE, ", ", Base.Sys.cpu_info()[1].model)
    println(io, ";; each row: a testset's own results (children separate)")
    println(io, "[", join(map(edn, rs), "\n "), "]")
end
tp = sum(r[:pass] for r in rs); tf = sum(r[:fail] for r in rs); te = sum(r[:error] for r in rs); tb = sum(r[:broken] for r in rs)
println("TOTAL pass $tp fail $tf error $te broken $tb in $(length(rs)) testsets, $(round(top.t1 - top.t0; digits=1)) s")
