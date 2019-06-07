using Test
using PooledArrays

let a = rand(10), b = rand(10,10), c = rand(1:10, 1000)
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

    @test eltype(PooledArray(rand(128)).refs) == UInt8
    @test eltype(PooledArray(rand(300)).refs) == UInt16
    @test PooledVector == PooledArray{T, R, 1} where {T, R}
    @test PooledMatrix == PooledArray{T, R, 2} where {T, R}

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
