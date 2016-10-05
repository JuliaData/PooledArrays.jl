using Base.Test
using PooledArrays

let a = rand(10), b = rand(10,10), c = rand(1:10, 1000)
    @test PooledArray(a) == a
    @test PooledArray(b) == b
    pc = PooledArray(c)
    @test pc == c
    @test convert(Array, pc) == c
    @test issorted(pc.pool)

    @test copy(pc) == pc

    @test isa(similar(pc), typeof(pc))

    @test reverse(pc) == reverse(c)

    @test pc[34:250] == c[34:250]

    @test sort(pc) == sort(c)
    @test sortperm(pc) == sortperm(c)

    push!(pc, -10)
    push!(c, -10)
    @test pc == c
    @test issorted(pc.pool)

    @test map(identity, pc) == pc
    @test map(x->2x, pc) == map(x->2x, c)
end
