using HybridModelling
using Test
using Random

@testset "SegmentedTimeSeries" begin
    # Create a dummy dataset
    Xtrain = rand(10, 100);
    tsteps = 1:100;

    # testing simple setting
    sts = SegmentedTimeSeries(Xtrain; segmentsize=2, batchsize=1);
    @test length(sts) == 50 
    x_batch = first(sts)
    @assert dropdims(x_batch, dims=ndims(x_batch)) == Xtrain[:, 1:2]

    # testing with tuples
    sts = SegmentedTimeSeries((Xtrain, tsteps); segmentsize=2, batchsize=1);
    @test length(sts) == 50
    x_batch, tsteps_batch = first(sts)
    @assert dropdims(tsteps_batch, dims=ndims(tsteps_batch)) == tsteps[1:2]
    @assert dropdims(x_batch, dims=ndims(x_batch)) == Xtrain[:, 1:2]

    # testing with batch size
    sts = SegmentedTimeSeries(Xtrain; segmentsize=2, batchsize=2);
    @test length(sts) == 25 # 100 tsteps, segmentsize 2, batchsize 2 => 100/2/2 = 25 batches
    x_batch = first(sts) # Trigger the first iteration to ensure it works
    @assert x_batch[:, :, 2] == Xtrain[:, 3:4]


    # testing shift
    sts = SegmentedTimeSeries(tsteps; segmentsize=2, batchsize=1, shift=1);
    @test length(sts) == 99 # 100 tsteps, segmentsize 2, batchsize 2 => 100/2/2 = 25 batches
    batched_ts = collect(sts)
    @assert dropdims(batched_ts[1], dims=ndims(batched_ts[1])) == tsteps[1:2]
    @assert dropdims(batched_ts[2], dims=ndims(batched_ts[2])) == tsteps[2:3]

    sts = SegmentedTimeSeries(tsteps; segmentsize=3, batchsize=1, shift=1);
    @test length(sts) == 98 # 100 tsteps, segmentsize 2, batchsize 2 => 100/2/2 = 25 batches

    batched_ts = collect(sts)
    @assert dropdims(batched_ts[1], dims=ndims(batched_ts[1])) == tsteps[1:3]
    @assert dropdims(batched_ts[2], dims=ndims(batched_ts[2])) == tsteps[2:4]

    sts = SegmentedTimeSeries(tsteps; segmentsize=3, batchsize=1, shift=2);
    batched_ts = collect(sts)
    @test length(last(batched_ts)[:,1]) == 3 # 100 tsteps, segmentsize 3, batchsize 1, shift 2 => (100-3)/2 + 1 = 49 batches

    sts = SegmentedTimeSeries(tsteps; segmentsize=3, batchsize=1, shift=2, partial_segment=true);
    batched_ts = collect(sts)
    @test length(last(batched_ts)[:, 1]) != 3 # 100 tsteps, segmentsize 3, batchsize 1, shift 2 => (100-3)/2 + 1 = 49 batches

    sts = SegmentedTimeSeries(Xtrain; segmentsize=2, shift = 1, batchsize=2, partial_batch=true);
    @test length(sts) == 50 # 100 tsteps, segmentsize 2, batchsize 2 => 100/2/2 = 25 batches
    batched_ts = collect(sts)
    @test size(first(batched_ts), 3) == 2 # Each batch should have 2 segments
    @test size(last(batched_ts), 3) == 1 # Last batch should have 1
    
    rng = Random.MersenneTwister(42)
    sts = SegmentedTimeSeries(tsteps; segmentsize=2, batchsize=1, shuffle=true, rng=rng);
    tsteps_batch = first(sts)
    @assert tsteps_batch[1] != tsteps[1:2]

    # tokenization
    sts = SegmentedTimeSeries(tsteps; segmentsize=2, batchsize=2);
    tokenized_sts = tokenize(sts)
    (token, tsteps_batch) = first(tokenized_sts)
    @test token == [1, 2]
end
