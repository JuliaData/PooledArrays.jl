using Base.Test
using PooledArrays

let a = rand(10), b = rand(10,10), c = rand(1:10, 1000)
    @test PooledArray(a) == a
    @test PooledArray(b) == b
    pc = PooledArray(c)
    @test pc == c
    @test convert(Array, pc) == c
    #@test issorted(pc.pool)

    @test copy(pc) == pc

    @test isa(similar(pc), typeof(pc))

    @test reverse(pc) == reverse(c)

    @test pc[34:250] == c[34:250]

    @test sort(pc) == sort(c)
    @test sortperm(pc) == sortperm(c)

    push!(pc, -10)
    push!(c, -10)
    @test pc == c
    @test maximum(values(pc.pool)) == 11
    @test maximum(keys(pc.revpool)) == 11

    #@test issorted(pc.pool)

    @test map(identity, pc) == pc
    @test map(x->2x, pc) == map(x->2x, c)

    # case where the outputs are one-to-many
    pa = PooledArray([1,2,3,4,5,6])
    @test map(isodd, pa) == [1,0,1,0,1,0]

    px = PooledArray(rand(128))
    py = PooledArray(rand(200))

    @test isa(vcat(px, py), PooledArray{Float64, UInt16})
end
