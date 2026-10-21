# EDN for the Gen.jl fixtures (run_suite.jl, generate.jl): symbols become
# keywords with - for _, the non-finite floats ##NaN and ##±Inf, and a map
# takes a line per entry.
edn(x::Nothing) = "nil"
edn(x::Bool) = x ? "true" : "false"
edn(x::Integer) = string(x)
edn(x::AbstractFloat) = isnan(x) ? "##NaN" : isinf(x) ? (x > 0 ? "##Inf" : "##-Inf") : repr(Float64(x))
edn(x::Symbol) = ":" * replace(string(x), "_" => "-")
edn(x::AbstractString) = repr(String(x))
edn(x::Union{AbstractVector,Tuple}) = "[" * join(map(edn, collect(x)), " ") * "]"
edn(x::AbstractDict) = "{" * join(["$(edn(k)) $(edn(v))" for (k, v) in x], "\n ") * "}"
