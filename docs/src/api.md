# API

```@meta
CollapsedDocStrings = true
```


## Dataloaders
```@docs
SegmentedTimeSeries
tokenize
tokens
```

## Parameter Layers
```@docs
ParameterLayer
NoConstraint
BoxConstraint
NamedTupleConstraint
```

## Models
```@docs
ODEModel
AnalyticModel
ARModel
```

## Initial Conditions
```@docs
ICLayer
```

## Training API
```@docs
train
InferICs
```

### `Lux.jl` training backend
```@docs
SGDBackend
```

### `Turing.jl` backend
```@docs
BayesianLayer
getpriors
MCSamplingBackend
```
