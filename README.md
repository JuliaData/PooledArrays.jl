[![Build Status](https://travis-ci.org/JuliaComputing/PooledArrays.jl.svg?branch=master)](https://travis-ci.org/JuliaComputing/PooledArrays.jl)

[![codecov.io](http://codecov.io/github/JuliaComputing/PooledArrays.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaComputing/PooledArrays.jl?branch=master)

# PooledArrays.jl
A pooled representation of arrays for purposes of compression when there are few unique elements.

If you don't require `setindex!` functionality, you might want to use [IndirectArrays](https://github.com/JuliaArrays/IndirectArrays.jl) instead.
