## Loss function
# Although we could overload `Lux.GenericLossFunction`
# See https://github.com/LuxDL/Lux.jl/blob/13045f78bb98c57081494a1fb8ed8e6dbf151bb8/src/helpers/losses.jl#L763
# this function checks for size and throws an error if size differing; we do not want this behaviour.
# We lose the efficiency of `Lux.GenericLossFunction`, but benefits should be minor over computational time of the ODEModels
import Lux: AbstractLossFunction, xlogy
@concrete struct LogMSELoss <: AbstractLossFunction
    epsilon
end

LogMSELoss() = LogMSELoss(eps(Float32))

function (l::LogMSELoss)(data, pred)
    T = eltype(data)
    data = max.(data, T(l.epsilon)) # we do not tolerate negative data
    pred = max.(pred, T(l.epsilon)) #
    if size(data) != size(pred) # preventing Zygote to crash
        return T(Inf)
    end

    return mean((log.(data) .- log.(pred)).^2)
    # if l isa Number # preventing any other reason for Zygote to crash
    #     return l
    # else 
    #     return Inf
    # end
end


# TODO: to revise
@concrete struct PoissonLoss <: AbstractLossFunction
    epsilon
end

PoissonLoss() = PoissonLoss(eps(Float32))

function (l::PoissonLoss)(data, pred)
    T = eltype(data)
    data = max.(data, T(l.epsilon)) # we do not tolerate negative data
    pred = max.(pred, T(l.epsilon)) #
    if size(data) != size(pred) # preventing Zygote to crash
        return T(Inf)
    end

    return mean((pred .- xlogy.(data, pred)))
    # if l isa Number # preventing any other reason for Zygote to crash
    #     return l
    # else 
    #     return Inf
    # end
end