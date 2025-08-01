using ConcreteStructs: @concrete

abstract type AbstractModelFeatures end

@concrete struct ModelFeatures{T}
    u0
    t0
    tsteps
end


"""
    LearnableICs(in_dims => out_dims; projection = Constraint(-Inf, Inf), init_values = nothing)


"""
struct LearnableICs <: AbstractICs 
    in_dims <: Union{IntegerType,Tuple{Vararg{IntegerType}}}
    out_dims <: IntegerType
    projection
    init_values
end


LearmableICs((in_dims, out_dims); projection = Constraint(-Inf, Inf), init_values = nothing) =
    LearnableICs(in_dims, out_dims, projection, init_values)