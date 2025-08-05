using HybridModelling
using Test
using Random

@testset "SegmentedTimeSeries" begin
    # Create a dummy dataset
    Xtrain = rand(10, 100);
    tsteps = 1:100;

    sts = SegmentedTimeSeries(Xtrain; segmentsize=2, batchsize=1);
    @test length(sts) == 50 # 100 tsteps, segmentsize 2, batchsize 2 => 100/2/2 = 25 batches
    x_batch = first(sts)
    @assert x_batch[1] == Xtrain[:, 1:2]

    sts = SegmentedTimeSeries((Xtrain, tsteps); segmentsize=2, batchsize=1);
    @test length(sts) == 50 # 100 tsteps, segmentsize 2, batchsize 2 => 100/2/2 = 25 batches
    x_batch, tsteps_batch = first(sts)
    @assert tsteps_batch[1] == tsteps[1:2]
    @assert x_batch[1] == Xtrain[:, 1:2]

    sts = SegmentedTimeSeries(Xtrain; segmentsize=2, batchsize=2);
    @test length(sts) == 25 # 100 tsteps, segmentsize 2, batchsize 2 => 100/2/2 = 25 batches
    x_batch = first(sts) # Trigger the first iteration to ensure it works
    @assert x_batch[1] == Xtrain[:,1:2]


    sts = SegmentedTimeSeries(tsteps; segmentsize=2, batchsize=1, shift=1);
    @test length(sts) == 99 # 100 tsteps, segmentsize 2, batchsize 2 => 100/2/2 = 25 batches

    batched_ts = collect(sts)
    @assert batched_ts[1][1] == tsteps[1:2]
    @assert batched_ts[2][1] == tsteps[2:3]

    sts = SegmentedTimeSeries(tsteps; segmentsize=3, batchsize=1, shift=1);
    @test length(sts) == 98 # 100 tsteps, segmentsize 2, batchsize 2 => 100/2/2 = 25 batches

    batched_ts = collect(sts)
    @assert batched_ts[1][1] == tsteps[1:3]
    @assert batched_ts[2][1] == tsteps[2:4]

    sts = SegmentedTimeSeries(tsteps; segmentsize=3, batchsize=1, shift=2);
    batched_ts = collect(sts)
    @test length(last(batched_ts)[1]) == 3 # 100 tsteps, segmentsize 3, batchsize 1, shift 2 => (100-3)/2 + 1 = 49 batches

    sts = SegmentedTimeSeries(tsteps; segmentsize=3, batchsize=1, shift=2, partial_segment=true);
    batched_ts = collect(sts)
    @test length(last(batched_ts)[1]) != 3 # 100 tsteps, segmentsize 3, batchsize 1, shift 2 => (100-3)/2 + 1 = 49 batches

    sts = SegmentedTimeSeries(Xtrain; segmentsize=2, shift = 1, batchsize=2, partial_batch=true);
    @test length(sts) == 50 # 100 tsteps, segmentsize 2, batchsize 2 => 100/2/2 = 25 batches
    batched_ts = collect(sts)
    length(first(batched_ts)) == 2 # Each batch should have 3 segments
    @test length(last(batched_ts)) == 1 # Last batch should have 2 
    
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
