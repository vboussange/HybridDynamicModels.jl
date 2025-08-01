using HybridModelling
using Test
using Random


@testset "SegmentedTimeSeries" begin
    # Create a dummy dataset
    Xtrain = rand(10, 100);
    tsteps = 1:100;

    sts = SegmentedTimeSeries(Xtrain, tsteps; segmentsize=2, batchsize=1);
    @test length(sts) == 50 # 100 tsteps, segmentsize 2, batchsize 2 => 100/2/2 = 25 batches
    x_batch, tsteps_batch = first(sts)
    @assert tsteps_batch[1] == tsteps[1:2]
    @assert x_batch[1] == Xtrain[:, 1:2]

    sts = SegmentedTimeSeries(Xtrain, tsteps; segmentsize=2, batchsize=2);
    @test length(sts) == 25 # 100 tsteps, segmentsize 2, batchsize 2 => 100/2/2 = 25 batches
    x_batch, tsteps_batch = first(sts) # Trigger the first iteration to ensure it works
    @assert tsteps_batch[1] == tsteps[1:2]
    @assert tsteps_batch[2] == tsteps[3:4]


    sts = SegmentedTimeSeries(Xtrain, tsteps; segmentsize=2, batchsize=1, shift=1);
    @test length(sts) == 99 # 100 tsteps, segmentsize 2, batchsize 2 => 100/2/2 = 25 batches

    batched_ts = collect(sts)
    @assert batched_ts[1][2][1] == tsteps[1:2]
    @assert batched_ts[2][2][1] == tsteps[2:3]

    sts = SegmentedTimeSeries(Xtrain, tsteps; segmentsize=3, batchsize=1, shift=1);
    @test length(sts) == 98 # 100 tsteps, segmentsize 2, batchsize 2 => 100/2/2 = 25 batches

    batched_ts = collect(sts)
    @assert batched_ts[1][2][1] == tsteps[1:3]
    @assert batched_ts[2][2][1] == tsteps[2:4]

    sts = SegmentedTimeSeries(Xtrain, tsteps; segmentsize=2, batchsize=1, shift=1);

    batched_ts = collect(sts)
    @assert batched_ts[1][2][1] == tsteps[1:3]
    @assert batched_ts[2][2][1] == tsteps[2:4]

    rng = Random.MersenneTwister(42)
    sts = SegmentedTimeSeries(Xtrain, tsteps; segmentsize=2, batchsize=1, shuffle=true, rng=rng);
    x_batch, tsteps_batch = first(sts)
    @assert tsteps_batch[1] != tsteps[1:2]
    @assert x_batch[1] == Xtrain[:, tsteps_batch[1]]
end
