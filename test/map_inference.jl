for signed in (true, false), compress in (true, false), len in (1, 100, 1000)
    x = PooledArray(fill(1, len), signed=true, compress=true)
    @inferred PooledVector{Int, Int, Vector{Int}} map(identity, x)
end
