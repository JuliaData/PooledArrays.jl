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

    @test eltype(PooledArray(rand(128)).refs) == UInt8
    @test eltype(PooledArray(rand(300)).refs) == UInt16
    @test PooledVector == PooledArray{T, R, 1} where {T, R}
    @test PooledMatrix == PooledArray{T, R, 2} where {T, R}
end
