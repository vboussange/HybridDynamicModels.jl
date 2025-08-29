## Loss function
# See
# https://github.com/LuxDL/Lux.jl/blob/13045f78bb98c57081494a1fb8ed8e6dbf151bb8/src/helpers/losses.jl#L763
function _log_mseloss(data, pred, epsilon = 1e-6)
    T = eltype(data)

    # we do not tolerate negative data
    data = max.(data, T(epsilon))
    pred = max.(pred, T(epsilon))

    # preventing optimization to crash
    if size(data) != size(pred)
        return T(Inf)
    end
    
    return (log.(data) .- log.(pred)) .^ 2
end

"""
    LogMSELoss(; agg=mean, epsilon=1e-6)

Computes the mean squared error (MSE) between the log of the predictions and
targets. Clips values less than `epsilon`.
"""
function LogMSELoss(; agg = mean, epsilon = 1e-6)
    return GenericLossFunction(
        Base.Fix2(((x, y), epsilon) -> _log_mseloss(x, y, epsilon), epsilon); agg)
end