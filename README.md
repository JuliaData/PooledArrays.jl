# PooledArrays.jl

[![CI](https://github.com/JuliaData/PooledArrays.jl/workflows/CI/badge.svg)](https://github.com/JuliaData/PooledArrays.jl/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/JuliaData/PooledArrays.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaData/PooledArrays.jl)
[![deps](https://juliahub.com/docs/PooledArrays/deps.svg)](https://juliahub.com/ui/Packages/PooledArrays/vi11X?t=2)
[![version](https://juliahub.com/docs/PooledArrays/version.svg)](https://juliahub.com/ui/Packages/PooledArrays/vi11X)
[![pkgeval](https://juliahub.com/docs/PooledArrays/pkgeval.svg)](https://juliahub.com/ui/Packages/PooledArrays/vi11X)


A pooled representation of arrays for purposes of compression when there are few unique elements.

**Installation**: at the Julia REPL, `import Pkg; Pkg.add("PooledArrays")`

**Usage**:

Working with `PooledArray` objects does not differ from working with general
`AbstractArray` objects, with two exceptions:
* If you hold mutable objects in `PooledArray` it is not allowed to modify them
  after they are stored in it.
* In multi-threaded context it is not safe to assign values that are not already
  present in a `PooledArray`'s pool from one thread while either reading or
  writing to the same array from another thread.

Keeping in mind these two restrictions, as a user, the only thing you need to
learn is how to create `PooledArray` objects. This is accomplished by passing
an `AbstractArray` to the `PooledArray` constructor:

```
julia> using PooledArrays

julia> PooledArray(["a" "b"; "c" "d"])
2×2 PooledMatrix{String, UInt32, Matrix{UInt32}}:
 "a"  "b"
 "c"  "d"
 ```

`PooledArray` performs compression by storing an array of reference integers and
a mapping from integers to its elements in a dictionary. In this way, if the
size of the reference integer is smaller than the size of the actual elements
the resulting `PooledArray` has a smaller memory footprint than the equivalent
`Array`. By default `UInt32` is used as a type of reference integers. However,
you can specify the reference integer type you want to use by passing it as a
second argument to the constructor. This is usually done when you know that you
will have only a few unique elements in the `PooledArray`.

```
julia> PooledArray(["a", "b", "c", "d"], UInt8)
4-element PooledVector{String, UInt8, Vector{UInt8}}:
 "a"
 "b"
 "c"
 "d"
 ```

Alternatively you can pass the `compress` and `signed` keyword arguments to the
`PooledArray` constructor to automatically select the reference integer type.
When you pass `compress=true` then the reference integer type is chosen to be
the smallest type that is large enough to hold all unique values in array. When
you pass `signed=true` the reference type is signed (by default it is unsigned).
```
julia> PooledArray(["a", "b", "c", "d"]; compress=true, signed=true)
4-element PooledVector{String, Int8, Vector{Int8}}:
 "a"
 "b"
 "c"
 "d"
```

**Maintenance**: PooledArrays is maintained collectively by the
[JuliaData collaborators](https://github.com/orgs/JuliaData/people).
Responsiveness to pull requests and issues can vary,
depending on the availability of key collaborators.

## Related Packages

- [IndirectArrays](https://github.com/JuliaArrays/IndirectArrays.jl) 
- [CategoricalArrays](https://github.com/JuliaData/CategoricalArrays.jl)
