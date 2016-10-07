[![Build Status](https://travis-ci.org/JuliaComputing/PooledArrays.jl.svg?branch=master)](https://travis-ci.org/JuliaComputing/PooledArrays.jl)

[![codecov.io](http://codecov.io/github/JuliaComputing/PooledArrays.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaComputing/PooledArrays.jl?branch=master)

# PooledArrays.jl
A pooled representation of arrays for purposes of compression when there are few unique elements.

This implementation is designed for elements with a total order. The pool of unique values is
maintained in sorted order, allowing efficient comparison and sorting based on integer IDs.

If this sorting behavior is not wanted, you might want to use [IndirectArrays](https://github.com/JuliaArrays/IndirectArrays.jl) instead.
