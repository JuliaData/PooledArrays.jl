using Test
using PooledArrays
using DataAPI: refarray, refvalue, refpool, invrefpool

if !isdefined(Base, :copy!)
    using Future: copy!
end

if Threads.nthreads() < 2
    @warn("Running with only one thread: correctness of parallel operations is not tested")
else
    @show Threads.nthreads()
end

@testset "PooledArrays" begin
    a = rand(10)
    b = rand(10,10)
    c = rand(1:10, 1000)

    @test PooledArray(a) == a
    @test PooledArray(b) == b
    pc = PooledArray(c)
    @test pc == c
    @test convert(Array, pc) == c
    #@test issorted(pc.invpool)

    @test copy(pc) == pc

    @test isa(similar(pc), typeof(pc))

    @test reverse(pc) == reverse(c)

    @test pc[34:250] == c[34:250]

    @test sort(pc) == sort(c)
    @test sortperm(pc) == sortperm(c)

    push!(pc, -10)
    push!(c, -10)
    @test pc == c
    @test maximum(values(pc.invpool)) == 11
    @test length(pc.pool) == 11

    pc2 = copy(pc)
    @test append!(pc2, [3, 1]) === pc2
    @test pc2 == [pc; 3; 1]

    pc2 = copy(pc)
    @test deleteat!(pc2, 2) === pc2
    @test pc2 == pc[[1; 3:end]]
    @test deleteat!(pc2, [2, 4]) === pc2
    @test pc2 == pc[[1; 4; 6:end]]

    #@test issorted(pc.invpool)

    @test map(identity, pc) == pc
    @test map(x->2x, pc) == map(x->2x, c)

    # case where the outputs are one-to-many
    pa = PooledArray([1,2,3,4,5,6])
    @test map(isodd, pa) == [true,false,true,false,true,false]

    px = PooledArray(rand(128))
    py = PooledArray(rand(200))

    @test isa(vcat(px, py), PooledArray{Float64, UInt16})
    @test vcat(px, Array(py)) == vcat(px, py)

    px2 = copy(px)
    resize!(px2, 100)
    @test px2 == px[1:100]

    px3 = PooledArray(px2)
    px3[1] = 0
    @test px2 !== px3
    @test px2 != px3

    @test findall(PooledArray([true,false,true])) == [1,3]

    @test PooledArray{Union{Int,Missing}}([1, 2]) isa PooledArray{Union{Int,Missing}}

    @test eltype(PooledArray(rand(128)).refs) == UInt32
    @test eltype(PooledArray(rand(300)).refs) == UInt32
    @test eltype(PooledArray(rand(128), UInt8).refs) == UInt8
    @test eltype(PooledArray(rand(300), UInt16).refs) == UInt16
    @test PooledVector == PooledArray{T, R, 1} where {T, R}
    @test PooledMatrix == PooledArray{T, R, 2} where {T, R}

    s = PooledArray(["a", "a", "b"])
    @test eltype(PooledArray(s).refs) == UInt32
    @test eltype(PooledArray(s, signed=true).refs) == Int32
    @test eltype(PooledArray(s, compress=true).refs) == UInt8
    @test eltype(PooledArray(s, signed=true, compress=true).refs) == Int8
    @test eltype(PooledArray(rand(300), signed=true, compress=true).refs) == Int16
    @test all(refarray(s) .== [1, 1, 2])
    for i in 1:3
        @test refvalue(s, refarray(s)[i]) == s[i]
    end
    @test refpool(s) == ["a", "b"]
    @test invrefpool(s) == Dict("a" => 1, "b" => 2)

    @testset "push!" begin
        xs = PooledArray([10, 20, 30])
        @test xs === push!(xs, -100)
        @test xs == [10, 20, 30, -100]
    end

    @testset "pushfirst!" begin
        ys = PooledArray([10, 20, 30])
        @test ys === pushfirst!(ys, -100)
        @test ys == [-100, 10, 20, 30]
    end

    v1 = PooledArray([1, 3, 2, 4])
    v2 = PooledArray(BigInt.([1, 3, 2, 4]))
    v3 = PooledArray(["a", "c", "b", "d"])

    @test PooledArrays.fast_sortable(v1) === v1
    @test isbitstype(eltype(PooledArrays.fast_sortable(v1)))
    Base.Order.Perm(Base.Order.Forward, v1).data === v1

    @test PooledArrays.fast_sortable(v2) == PooledArray([1, 3, 2, 4])
    @test isbitstype(eltype(PooledArrays.fast_sortable(v2)))
    Base.Order.Perm(Base.Order.Forward, v2).data == PooledArray([1, 3, 2, 4])

    @test PooledArrays.fast_sortable(v3) == PooledArray([1, 3, 2, 4])
    @test isbitstype(eltype(PooledArrays.fast_sortable(v3)))
    Base.Order.Perm(Base.Order.Forward, v3).data == PooledArray([1, 3, 2, 4])
end

@testset "pool non-copying constructor and copy tests" begin
    pa = PooledArray([1, 2, 3])
    @test pa.refcount[] == 1
    pa2 = PooledArray(pa)
    @test pa.refcount[] == 2
    @test pa.refs == pa2.refs
    @test pa.refs !== pa2.refs
    @test pa.pool === pa2.pool
    @test pa.invpool === pa2.invpool
    @test pa.refcount === pa2.refcount

    pav = @view pa[[3, 1]]

    @test pav == [3, 1]
    @test DataAPI.refarray(pav) isa SubArray{UInt32,1,Array{UInt32,1},Tuple{Array{Int,1}},false}
    @test DataAPI.refpool(pav) === pa.pool
    @test DataAPI.invrefpool(pav) === pa.invpool
    @test_throws BoundsError DataAPI.refvalue(pav, 0)
    @test DataAPI.refvalue(pav, 1) === 1
    @test DataAPI.refvalue(pav, 2) === 2
    @test DataAPI.refvalue(pav, 3) === 3
    @test_throws BoundsError DataAPI.refvalue(pav, 4)

    @test pa.refcount[] == 2
    pa3 = PooledArray(pav)
    @test pa.refcount[] == 3
    @test pa.refs[[3, 1]] == pa3.refs
    @test pa.refs !== pa3.refs
    @test pa.pool === pa3.pool
    @test pa.invpool === pa3.invpool
    @test pa.refcount === pa3.refcount
    pa2 = pa3
    # try to force GC to check finalizer
    GC.gc(); GC.gc(); GC.gc(); GC.gc()
    @test pa.refcount[] == 2

    pa = PooledArray([1, 2, 3])
    @test pa.refcount[] == 1
    pa2 = copy(pa)
    @test pa.refcount[] == 2
    @test pa.refs == pa2.refs
    @test pa.refs !== pa2.refs
    @test pa.pool === pa2.pool
    @test pa.invpool === pa2.invpool
    @test pa.refcount === pa2.refcount

    pav = @view pa[1]
    pa3 = copy(pav)
    @test pa.refcount[] == 3
    @test DataAPI.refarray(pav) == pa3.refs
    @test pa.refs !== pa3.refs
    @test pa.pool === pa3.pool
    @test pa.invpool === pa3.invpool
    @test pa.refcount === pa3.refcount
end

@testset "test de-referencing on setindex!" begin
    pa = PooledArray([1, 2, 3])
    @test pa.refcount[] == 1
    pa2 = copy(pa)
    @test pa.refcount[] == 2
    old_pool = pa.pool
    old_invpool = pa.invpool
    old_refcount = pa.refcount

    # within pool
    pa2[1] = 3
    @test pa == 1:3
    @test pa.pool === old_pool
    @test pa.invpool === old_invpool
    @test pa.refcount === old_refcount
    @test pa.refcount[] == 2
    @test pa2 == [3, 2, 3]
    @test pa2.pool === old_pool
    @test pa2.invpool === old_invpool
    @test pa2.refcount === old_refcount

    # new value
    pa2[1] = 4
    @test pa == 1:3
    @test pa.pool == old_pool
    @test pa.invpool == old_invpool
    @test pa.refcount == old_refcount
    @test pa.refcount[] == 1
    @test pa2 == [4, 2, 3]
    @test pa2.pool !== old_pool
    @test pa2.invpool !== old_invpool
    @test pa2.refcount !== old_refcount
    @test pa2.refcount[] == 1
end

@testset "copy! tests" begin
    pa = PooledArray([1 2; 3 4])
    pav = @view pa[1:2]
    pac = @view pa[1:2]

    pa2 = PooledArray([1])
    pa3 = PooledArray([1 2 3; 4 5 6])

end
