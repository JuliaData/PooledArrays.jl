using Test, OffsetArrays
using PooledArrays
using DataAPI: refarray, refvalue, refpool, invrefpool
using PooledArrays: refcount
using Random: randperm

import Future.copy!

if Threads.nthreads() < 2
    @warn("Running with only one thread: correctness of parallel operations is not tested")
else
    @show Threads.nthreads()
end

@testset "PooledArrays" begin
    @test eltype(PooledArray(Int[])) === Int

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

    perm = randperm(length(pc))
    pc2 = copy(pc)
    pc3 = permute!(pc2, perm)
    @test pc2 === pc3
    @test pc2 == pc[perm]

    pc2 = copy(pc)
    pc3 = invpermute!(pc2, perm)
    @test pc2 === pc3
    @test pc2[perm] == pc

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

    for T in (Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64)
        @inferred PooledArray([1, 2, 3], T)
        @inferred PooledArray([1 2 3], T)
    end
    @test typeof(PooledArray([1 2; 3 4], Int8)) === PooledMatrix{Int, Int8, Matrix{Int8}}

    for signed in (true, false), compress in (true, false)
        @test_throws ErrorException @inferred PooledArray([1, 2, 3],
                                                          signed=signed,
                                                          compress=compress)
    end
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
    @test refarray(pav) isa SubArray{UInt32,1,Array{UInt32,1},Tuple{Array{Int,1}},false}
    @test refpool(pav) === pa.pool
    @test invrefpool(pav) === pa.invpool
    @test_throws BoundsError refvalue(pav, 0)
    @test refvalue(pav, 1) === 1
    @test refvalue(pav, 2) === 2
    @test refvalue(pav, 3) === 3
    @test_throws BoundsError refvalue(pav, 4)

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
    if pa.refcount[] != 2
        @warn "finalizer of PooledArray not triggered; excess refs: $(pa.refcount[] - 2)"
    end

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
    @test pa3 isa PooledArray{Int, UInt32, 0}
    @test pa.refcount[] == 3
    @test refarray(pav) == pa3.refs
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
    pa = PooledArray(1:4)
    pa1 = pa[1:2]
    pav1 = @view pa[2:3]
    pa2 = PooledArray(fill(2))
    pav2 = @view pa[3]

    pat1 = PooledArray([0, 0])
    copy!(pat1, pa1)
    @test pat1 == pa1
    @test refpool(pat1) === refpool(pa1)
    @test invrefpool(pat1) === invrefpool(pa1)
    @test refcount(pat1) === refcount(pa1)
    @test refcount(pat1)[] == 3

    copy!(pat1, pav1)
    @test pat1 == pav1
    @test refpool(pat1) === refpool(pav1)
    @test invrefpool(pat1) === invrefpool(pav1)
    @test refcount(pat1) === refcount(pav1)
    @test refcount(pat1)[] == 3

    pat2 = PooledArray(fill(0))
    copy!(pat2, pa2)
    @test pat2 == pa2
    @test refpool(pat2) === refpool(pa2)
    @test invrefpool(pat2) === invrefpool(pa2)
    @test refcount(pat2) === refcount(pa2)
    @test refcount(pat2)[] == 2

    copy!(pat2, pav2)
    @test pat2 == pav2
    @test refpool(pat2) === refpool(pav2)
    @test invrefpool(pat2) === invrefpool(pav2)
    @test refcount(pat2) === refcount(pav2)
    @test refcount(pat2)[] == 4
    @test refcount(pa2)[] == 1
end

@testset "correct refcount when treading" begin
    pa = PooledArray([1 2; 3 4])
    x = Vector{Any}(undef, 120)
    Threads.@threads for i in 1:120
        x[i] = copy(pa)
    end
    @test pa.refcount[] == 121
    Threads.@threads for i in 1:61
        @test x[i].refcount === pa.refcount
        x[i][1] = 2
        @test x[i].refcount === pa.refcount
        x[i][1] = 5
        @test x[i].refcount[] == 1
    end
    @test pa.refcount[] == 60
    x = nothing
    # try to force GC to check finalizer
    GC.gc(); GC.gc(); GC.gc(); GC.gc()
    if pa.refcount[] != 1
        @warn "finalizer of PooledArray not triggered; excess refs: $(pa.refcount[] - 1)"
    end
end

@testset "copyto! tests" begin
    pa1 = PooledArray([1, 2, 3])
    pa2 = similar(pa1, 4)
    @test_throws BoundsError copyto!(pa1, pa2)
    copyto!(pa2, pa1)
    @test refpool(pa2) === refpool(pa1)
    @test invrefpool(pa2) === invrefpool(pa1)
    @test refcount(pa2) === refcount(pa1)
    @test refcount(pa2)[] == 2

    pa1 = view(PooledArray([1, 2, 3]), :)
    pa2 = similar(pa1, 4)
    @test_throws BoundsError copyto!(pa1, pa2)
    copyto!(pa2, pa1)
    @test refpool(pa2) === refpool(pa1)
    @test invrefpool(pa2) === invrefpool(pa1)
    @test refcount(pa2) === refcount(pa1)
    @test refcount(pa2)[] == 2

    pa1 = PooledArray([1, 2, 3])
    pa2 = view(similar(pa1, 4), :)
    @test_throws BoundsError copyto!(pa1, pa2)
    copyto!(pa2, pa1)
    @test refpool(pa2) === refpool(pa1)
    @test invrefpool(pa2) === invrefpool(pa1)
    @test refcount(pa2) === refcount(pa1)
    @test refcount(pa2)[] == 2

    pa1 = view(PooledArray([1, 2, 3]), :)
    pa2 = view(similar(pa1, 4), :)
    @test_throws BoundsError copyto!(pa1, pa2)
    copyto!(pa2, pa1)
    @test refpool(pa2) === refpool(pa1)
    @test invrefpool(pa2) === invrefpool(pa1)
    @test refcount(pa2) === refcount(pa1)
    @test refcount(pa2)[] == 2

    pa1 = PooledArray([1, 2, 3])
    pa2 = similar(pa1, 4)
    copyto!(pa2, 1, view(pa1, [1, 1]), 1, 2)
    @test refpool(pa2) === refpool(pa1)
    @test invrefpool(pa2) === invrefpool(pa1)
    @test refcount(pa2) === refcount(pa1)
    @test refcount(pa2)[] == 2
    copyto!(pa2, 3, view(pa1, [1, 1]), 1, 2)
    @test refpool(pa2) === refpool(pa1)
    @test invrefpool(pa2) === invrefpool(pa1)
    @test refcount(pa2) === refcount(pa1)
    @test refcount(pa2)[] == 2
    @test pa2 == [1, 1, 1, 1]

    pa1 = PooledArray([1, 2, 3])
    pa2 = similar(pa1, Float64, 3)
    copyto!(pa2, 1, pa1, 1, 3)
    @test refpool(pa2) !== refpool(pa1)
    @test invrefpool(pa2) !== invrefpool(pa1)
    @test refcount(pa2) !== refcount(pa1)
    @test refcount(pa1)[] == 1
    @test refcount(pa2)[] == 1
    @test pa2 == [1, 2, 3]

    pa1 = PooledArray([1, 2, 3])
    pa2 = similar(pa1, 3)
    pa2 .= 1
    copyto!(pa2, 1, pa1, 1, 3)
    @test refpool(pa2) !== refpool(pa1)
    @test invrefpool(pa2) !== invrefpool(pa1)
    @test refcount(pa2) !== refcount(pa1)
    @test refcount(pa1)[] == 1
    @test refcount(pa2)[] == 1
    @test pa2 == [1, 2, 3]
end

@testset "reverse" begin
    pa1 = PooledArray([1, 2, 3])
    pa2 = reverse(pa1)
    pa3 = reverse(pa2)
    @test pa2 == [3, 2, 1]
    @test pa3 == pa1
    @test refpool(pa1) === refpool(pa2) === refpool(pa3)
    @test invrefpool(pa1) === invrefpool(pa2) === invrefpool(pa3)
    @test refcount(pa1) === refcount(pa2) === refcount(pa3)
    @test refcount(pa1)[] == 3
end

@testset "convert" begin
    pa1 = PooledArray([1, 2, 3])
    @test convert(PooledArray, pa1) === pa1
    @test eltype(convert(PooledArray{Float64}, pa1)) === Float64
    pa1c = convert(PooledArray{Int, UInt64, 1}, pa1)
    @test pa1c isa PooledArray{Int,UInt64,1,Array{UInt64,1}}
    @test pa1c == pa1
    @test !(pa1c isa typeof(pa1))
end

@testset "indexing" begin
    pa = PooledArray([1 2; 3 4])
    @test pa[2, 2, 1] == 4
    @test pa[2, 2] == 4
    @test pa[big(2), 2] == 4
    pav = view(pa, :, :)
    @test pav[2, 2, 1] == 4
    @test pav[2, 2] == 4
    @test pav[big(2), 2] == 4

    @test refcount(pa)[] == 1
    pa2 = pa[[true, false, false, true]]
    @test pa2 == [1, 4]
    @test refpool(pa) === refpool(pa2)
    @test invrefpool(pa) === invrefpool(pa2)
    @test refcount(pa) === refcount(pa2)
    @test refcount(pa)[] == 2
    pa3 = pav[[true, false, false, true]]
    @test pa3 == [1, 4]
    @test refpool(pa) === refpool(pa3)
    @test invrefpool(pa) === invrefpool(pa3)
    @test refcount(pa) === refcount(pa3)
    @test refcount(pa)[] == 3

    # these checks are mostly needed to check for dispatch ambiguities
    @test pa[1] == 1
    @test pa[1, 1] == 1
    @test pa[1, 1, 1] == 1
    @test pa[:] == [1, 3, 2, 4]
    @test pa[1:4] == [1, 3, 2, 4]
    @test pa[collect(1:4)] == [1, 3, 2, 4]
    @test pa[1, 1:2] == [1, 2]
    @test pa[1, [1, 2]] == [1, 2]
    @test pa[1:1, 1:2] == [1 2]
    @test pa[1:1, [1, 2]] == [1 2]
    @test pa[[1], 1:2] == [1 2]
    @test pa[[1], [1, 2]] == [1 2]
    @test pav[1] == 1
    @test pav[1, 1] == 1
    @test pav[1, 1, 1] == 1
    @test pav[:] == [1, 3, 2, 4]
    @test pav[1:4] == [1, 3, 2, 4]
    @test pav[collect(1:4)] == [1, 3, 2, 4]
    @test pav[1, 1:2] == [1, 2]
    @test pav[1, [1, 2]] == [1, 2]
    @test pav[1:1, 1:2] == [1 2]
    @test pav[1:1, [1, 2]] == [1 2]
    @test pav[[1], 1:2] == [1 2]
    @test pav[[1], [1, 2]] == [1 2]

    pav2 = view(PooledArray([1]), 1)
    pa2 = similar(pav2)
    pa2[] = 10

    @test pav2[] == 1
    @test pa2[] == 10
end

@testset "isassigned" begin
    pa1 = PooledArray(["a"])
    pa2 = similar(pa1, 2)
    pa2v = view(pa2, 1)
    @test !isassigned(pa2, 1)
    @test !isassigned(pa2v)
end

@testset "setindex!" begin
    pa = PooledArray([1 2; 3 4])
    pa[1, 1] = 10
    @test pa == [10 2; 3 4]
    @test [pa pa] == [10 2 10 2; 3 4 3 4]
    pa[2] = 1000
    @test pa == [10 2; 1000 4]
    pa[1, :] = [11, 12]
    @test pa == [11 12; 1000 4]
    pa[1:2, 1:1] = [111, 222]
    @test pa == [111 12; 222 4]
    pa[1, 1, 1] = 0
    @test pa == [0 12; 222 4]
    pa[:] = [1 2; 3 4]
    @test pa == [1 2; 3 4]
end

@testset "repeat" begin
    pa1 = PooledArray([1, 2, 3])
    pa2 = repeat(pa1)
    pa3 = repeat(pa1, 2)
    pa4 = repeat(pa1, inner = 2)
    @test pa2 == pa1
    @test pa3 == [1, 2, 3, 1, 2, 3]
    @test pa4 == [1, 1, 2, 2, 3, 3]
    @test refpool(pa1) === refpool(pa2) === refpool(pa3) === refpool(pa4)
    @test invrefpool(pa1) === invrefpool(pa2) === invrefpool(pa3) === invrefpool(pa4)
    @test refcount(pa1) === refcount(pa2) === refcount(pa3) === refcount(pa4)
    @test refcount(pa1)[] == 4

    pa1 = PooledArray(["one", "two"])
    pa2 = repeat(pa1, outer = 3)
    pa3 = repeat(pa1, inner = 3, outer = 2)
    @test pa2 == ["one", "two", "one", "two", "one", "two"]
    @test pa3 == ["one", "one", "one", "two", "two", "two", "one", "one", "one", "two", "two", "two"]

    # missing shouldn't be a problem

    pa1 = PooledArray(["one", missing, "two"])
    pa2 = repeat(pa1, 2)
    pa3 = repeat(pa1, 0)
    @test isequal(pa2, ["one", missing, "two", "one", missing, "two"])
    @test isequal(pa2.pool, ["one", missing, "two"])
    @test size(pa3) == (0,)
    @test isempty(pa3.refs)

    # two dimensional
    pa1 = PooledArray([true false; false true; true true])
    pa2 = repeat(pa1, 2)
    @test pa2 == Bool[1 0; 0 1; 1 1; 1 0; 0 1; 1 1]

    pa1 = PooledArray([1 2; 3 4])
    pa2 = repeat(pa1, inner = (2, 1))
    @test pa2 == [1 2; 1 2; 3 4; 3 4]
end

@testset "map pure tests" begin
    x = PooledArray([1, 2, 3])
    x[3] = 1
    y = map(-, x, pure=true)
    @test refpool(y) == [-1, -2, -3]
    @test y == [-1, -2, -1]

    y = map(-, x)
    @test refpool(y) == [-1, -2]
    @test y == [-1, -2, -1]

    function f()
        i = Ref(0)
        return x -> (i[] -= 1; i[])
    end

    # the order is strange as we iterate invpool which is a Dict
    # and it depends on the version of Julia
    y = map(f(), x, pure=true)
    d = Dict(Set(1:3) .=> -1:-1:-3)
    @test refpool(y) == [d[i] for i in 1:3]
    @test y == [d[v] for v in x]

    y = map(f(), x)
    @test refpool(y) == [-1, -2, -3]
    @test y == [-1, -2, -3]

    x = PooledArray([1, missing, 2])
    y = map(identity, x)
    @test isequal(y, [1, missing, 2])
    @test typeof(y) === PooledVector{Union{Missing, Int}, UInt32, Vector{UInt32}}

    x = PooledArray([1, missing, 2], signed=true, compress=true)
    y = map(identity, x)
    @test isequal(y, [1, missing, 2])
    @test typeof(y) === PooledVector{Union{Missing, Int}, Int8, Vector{Int8}}

    x = PooledArray(fill(1, 200), signed=true, compress=true)
    y = map(f(), x)
    @test y == -1:-1:-200
    @test typeof(y) === PooledVector{Int, Int, Vector{Int}}

    x = PooledArray(reshape(fill(1, 200), 2, :), signed=true, compress=true)
    y = map(f(), x)
    @test y == reshape(-1:-1:-200, 2, :)
    @test typeof(y) === PooledMatrix{Int, Int, Matrix{Int}}

    x = PooledArray(fill("a"), signed=true, compress=true)
    y = map(f(), x)
    @test y == fill(-1)
    @test typeof(y) === PooledArray{Int, Int8, 0, Array{Int8, 0}}

    @static if VERSION >= v"1.6"
        for signed in (true, false), compress in (true, false), len in (1, 100, 1000)
            x = PooledArray(fill(1, len), signed=signed, compress=compress)
            @inferred PooledVector{Int, Int, Vector{Int}} map(identity, x)
        end
    end
end

@testset "insert! test" begin
    x = PooledArray([1, 2, 3])
    @test insert!(x, 2, 10) == [1, 10, 2, 3]
    @test_throws ArgumentError insert!(x, true, 10)
    @test_throws BoundsError insert!(x, 0, 10)
    @test x == [1, 10, 2, 3]
    @test x.pool == [1, 2, 3, 10]
    @test insert!(x, 3, 'c') == [1, 10, 99, 2, 3]
    @test x.pool == [1, 2, 3, 10, 99]
    @test insert!(x, 1, true) == [1, 1, 10, 99, 2, 3]
    @test x.pool == [1, 2, 3, 10, 99]
    @test insert!(x, 7, true) == [1, 1, 10, 99, 2, 3, 1]
    @test_throws BoundsError insert!(x, 9, true)
    @test x == [1, 1, 10, 99, 2, 3, 1]
end

@testset "pop! and popfirst!" begin
    x = PooledArray([1, 2, 3])
    @test pop!(x) == 3
    @test x == [1, 2]
    @test popfirst!(x) == 1
    @test x == [2]
    x = PooledArray(["1", "2", "3"])
    @test pop!(x) == "3"
    @test x == ["1", "2"]
    @test popfirst!(x) == "1"
    @test x == ["2"]
end

@testset "constructor corner cases" begin
    x = Vector{Any}(undef, 3)
    y = PooledArray(x)
    @test y isa PooledArray{Any}
    @test !any(i -> isassigned(y, i), eachindex(y))
    @test all(iszero, y.refs)
    @test isempty(y.pool)
    @test isempty(y.invpool)

    x[2] = "a"
    for v in (x, OffsetVector(x, -5))
        y = PooledArray(v)
        @test y isa PooledArray{Any}
        @test !isassigned(x, 1)
        @test x[2] == "a"
        @test !isassigned(x, 3)
        @test y.refs == [0, 1, 0]
        @test y.pool == ["a"]
        @test y.invpool == Dict("a" => 1)
    end
end
