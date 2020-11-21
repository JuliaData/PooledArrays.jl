module PooledArrays

import DataAPI

export PooledArray, PooledVector, PooledMatrix, dropunusedlevels!

##############################################################################
##
## PooledArray type definition
##
##############################################################################

const DEFAULT_POOLED_REF_TYPE = UInt32

# This is used as a wrapper during PooledArray construction only, to distinguish
# arrays of pool indices from normal arrays
mutable struct RefArray{R}
    a::R
end

function _invert(d::Dict{K, V}) where {K, V}
    d1 = Vector{K}(undef, length(d))
    for (k, v) in d
        d1[v] = k
    end
    return d1
end

mutable struct PooledArray{T, R <: Integer, N, RA} <: AbstractArray{T, N}
    refs::RA
    pool::Vector{T}
    invpool::Dict{T, R}

    function PooledArray(
        rs::RefArray{RA},
        invpool::Dict{T, R},
        pool = _invert(invpool),
    ) where {T, R, N, RA <: AbstractArray{R, N}}
        # refs mustn't overflow pool
        if length(rs.a) > 0 && maximum(rs.a) > length(invpool)
            throw(ArgumentError("Reference array points beyond the end of the pool"))
        end
        return new{T, R, N, RA}(rs.a, pool, invpool)
    end

    function PooledArray(
        rs::RefArray{RA},
        pool::Vector{T},
        invpool::Dict{T, R}=Dict{T, R}(x => i for (i, x) in enumerate(pool))
    ) where {T, R, N, RA <: AbstractArray{R, N}}
        # refs mustn't overflow pool
        if length(rs.a) > 0 && maximum(rs.a) > length(invpool)
            throw(ArgumentError("Reference array points beyond the end of the pool"))
        end
        return new{T, R, N, RA}(rs.a, pool, invpool)
    end

end
const PooledVector{T, R} = PooledArray{T, R, 1}
const PooledMatrix{T, R} = PooledArray{T, R, 2}

##############################################################################
##
## PooledArray constructors
##
# Algorithm:
# * Start with:
#   * A null pool
#   * A pre-allocated refs
#   * A hash from T to Int
# * Iterate over d
#   * If value of d in pool already, set the refs accordingly
#   * If value is new, add it to the pool, then set refs
##############################################################################

# Echo inner constructor as an outer constructor
function PooledArray(
    refs::RefArray{R},
    invpool::Dict{T, R},
    pool = _invert(invpool),
) where {T, R}
    return PooledArray{T, eltype(R), ndims(R), R}(refs, invpool, pool)
end

PooledArray(d::PooledArray) = copy(d)

function _label(
    xs::AbstractArray,
    ::Type{T} = eltype(xs),
    ::Type{I} = DEFAULT_POOLED_REF_TYPE,
    start = 1,
    labels = Array{I}(undef, size(xs)),
    invpool::Dict{T, I} = Dict{T, I}(),
    pool::Vector{T} = T[],
    nlabels = 0,
) where {T, I <: Integer}
    @inbounds for i in start:length(xs)
        x = xs[i]
        lbl = get(invpool, x, zero(I))
        if lbl !== zero(I)
            labels[i] = lbl
        else
            if nlabels == typemax(I)
                I2 = _widen(I)
                return _label(
                    xs,
                    T,
                    I2,
                    i,
                    convert(Vector{I2}, labels),
                    convert(Dict{T, I2}, invpool),
                    pool,
                    nlabels,
                )
            end
            nlabels += 1
            labels[i] = nlabels
            invpool[x] = nlabels
            push!(pool, x)
        end
    end
    return labels, invpool, pool
end

_widen(::Type{UInt8}) = UInt16
_widen(::Type{UInt16}) = UInt32
_widen(::Type{UInt32}) = UInt64

# Constructor from array, invpool, and ref type

"""
    PooledArray(array, [reftype])

Freshly allocate `PooledArray` using the given array as a source where each
element will be referenced as an integer of the given type.
If no `reftype` is specified one is chosen automatically based on the number of unique elements.
If `array` is not a `PooledArray` then the order of elements in `refpool` in the resulting
`PooledArray` is the order of first appereance of elements in `array`.
"""
PooledArray

function PooledArray{T}(d::AbstractArray, r::Type{R}) where {T, R <: Integer}
    refs, invpool, pool = _label(d, T, R)

    if length(invpool) > typemax(R)
        throw(ArgumentError("Cannot construct a PooledArray with type $R with a pool of size $(length(pool))"))
    end

    # Assertions are needed since _label is not type stable
    return PooledArray(RefArray(refs::Vector{R}), invpool::Dict{T, R}, pool)
end

function PooledArray{T}(d::AbstractArray) where {T}
    refs, invpool, pool = _label(d, T)
    return PooledArray(RefArray(refs), invpool, pool)
end

PooledArray(d::AbstractArray{T}, r::Type) where {T} = PooledArray{T}(d, r)
PooledArray(d::AbstractArray{T}) where {T} = PooledArray{T}(d)

# Construct an empty PooledVector of a specific type
PooledArray(t::Type) = PooledArray(Array(t, 0))
PooledArray(t::Type, r::Type) = PooledArray(Array(t, 0), r)

##############################################################################
##
## Basic interface functions
##
##############################################################################

DataAPI.refarray(pa::PooledArray) = pa.refs
DataAPI.refvalue(pa::PooledArray, i::Integer) = pa.pool[i]
DataAPI.refpool(pa::PooledArray) = pa.pool

Base.size(pa::PooledArray) = size(pa.refs)
Base.length(pa::PooledArray) = length(pa.refs)
Base.lastindex(pa::PooledArray) = lastindex(pa.refs)

Base.copy(pa::PooledArray) = PooledArray(RefArray(copy(pa.refs)), copy(pa.invpool))
# TODO: Implement copy_to()

function Base.resize!(pa::PooledArray{T, R, 1}, n::Integer) where {T, R}
    oldn = length(pa.refs)
    resize!(pa.refs, n)
    pa.refs[(oldn + 1):n] .= zero(R)
    return pa
end

Base.reverse(x::PooledArray) = PooledArray(RefArray(reverse(x.refs)), x.invpool)

function Base.permute!!(x::PooledArray, p::AbstractVector{T}) where {T <: Integer}
    Base.permute!!(x.refs, p)
    return x
end

function Base.invpermute!!(x::PooledArray, p::AbstractVector{T}) where {T <: Integer}
    Base.invpermute!!(x.refs, p)
    return x
end

function Base.similar(pa::PooledArray{T, R}, S::Type, dims::Dims) where {T, R}
    return PooledArray(RefArray(zeros(R, dims)), Dict{S, R}())
end

Base.findall(pdv::PooledVector{Bool}) = findall(convert(Vector{Bool}, pdv))

##############################################################################
##
## map
## Calls `f` only once per pool entry.
##
##############################################################################

"""
    dropunusedlevels!(x::PooledArray)

When PooledArrays are modified, via `setindex!`, or deleting elements,
pooled values that were present originally may no longer appear in the
array, yet still exist in the "pool". This can be desirable if these
values may still be reintroduced, but sometimes, it's desirable to
completely drop any reference to these pooled values. This need
may come up when calling `map` on a modified PooledArray, which
only applies the mapping function to the pool as an optimization.
This can lead to unexpected results if the mapping function
encounters pooled values that may not be present in the current array.
"""
function dropunusedlevels!(x::PooledArray{T, R}) where {T, R}
    # count number of occurences of each pool value
    counts = zeros(Int, length(x.pool))
    for ref in x.refs
        @inbounds counts[ref] += 1
    end
    if any(iszero, counts)
        # if there are any 0 counts, remove the pool value
        # and prepare to recode any higher refs
        newpoolidx = zeros(Int, length(x.pool))
        numtoremove = 0
        for (i, count) in enumerate(counts)
            if count == 0
                numtoremove += 1
            else
                newpoolidx[i] = i - numtoremove
            end
        end
        # recode existing refs with adjusted values
        for (i, val) in enumerate(x.refs)
            @inbounds x.refs[i] = R(newpoolidx[val])
        end
        # now to adjust pool, remove unused and recode invpool
        for i = length(counts):-1:1
            if counts[i] == 0
                val = x.pool[i]
                delete!(x.invpool, val)
                deleteat!(x.pool, i)
            else
                x.invpool[x.pool[i]] = R(newpoolidx[i])
            end
        end
    end
    return
end

function Base.map(f, x::PooledArray{T, R}; dropunusedlevels::Bool=false) where {T, R <: Integer}
    if dropunusedlevels
        dropunusedlevels!(x)
    end
    newpool = map(x.pool) do val
        try
            f(val)
        catch e
            @warn "error applying `f` to $val; you may need to call `dropunusedlevels!(x)` first"
            rethrow(e)
        end
    end
    return PooledArray(RefArray(copy(x.refs)), newpool)
end

##############################################################################
##
## Sorting can use the pool to speed things up
##
##############################################################################

function groupsort_indexer(x::AbstractVector, ngroups::Integer, perm)
    # translated from Wes McKinney's groupsort_indexer in pandas (file: src/groupby.pyx).

    # count group sizes, location 0 for NA
    n = length(x)
    # counts = x.invpool
    counts = fill(0, ngroups + 1)
    @inbounds for i in 1:n
        counts[x[i] + 1] += 1
    end
    counts[2:end] = counts[perm .+ 1]

    # mark the start of each contiguous group of like-indexed data
    where = fill(1, ngroups + 1)
    @inbounds for i in 2:(ngroups + 1)
        where[i] = where[i - 1] + counts[i - 1]
    end

    # this is our indexer
    result = fill(0, n)
    iperm = invperm(perm)

    @inbounds for i in 1:n
        label = iperm[x[i]] + 1
        result[where[label]] = i
        where[label] += 1
    end
    return result, where, counts
end

function Base.sortperm(
    pa::PooledArray;
    alg::Base.Sort.Algorithm = Base.Sort.DEFAULT_UNSTABLE,
    lt::Function = isless,
    by::Function = identity,
    rev::Bool = false,
    order = Base.Sort.Forward,
    _ord = Base.ord(lt, by, rev, order),
    poolperm = sortperm(pa.pool, alg = alg, order = _ord),
)
    return groupsort_indexer(pa.refs, length(pa.pool), poolperm)[1]
end

Base.sort(pa::PooledArray; kw...) = pa[sortperm(pa; kw...)]

#type FastPerm{O<:Base.Sort.Ordering,V<:AbstractVector} <: Base.Sort.Ordering
#    ord::O
#    vec::V
#end
#Base.sortperm{V}(x::AbstractVector, a::Base.Sort.Algorithm, o::FastPerm{Base.Sort.ForwardOrdering,V}) = x[sortperm(o.vec)]
#Base.sortperm{V}(x::AbstractVector, a::Base.Sort.Algorithm, o::FastPerm{Base.Sort.ReverseOrdering,V}) = x[reverse(sortperm(o.vec))]
#Perm{O<:Base.Sort.Ordering}(o::O, v::PooledVector) = FastPerm(o, v)

##############################################################################
##
## conversions
##
##############################################################################

Base.convert(
    ::Type{PooledArray{S, R1, N}},
    pa::PooledArray{T, R2, N},
) where {S, T, R1 <: Integer, R2 <: Integer, N} =
    PooledArray(RefArray(convert(Array{R1, N}, pa.refs)), convert(Dict{S, R1}, pa.invpool))
Base.convert(
    ::Type{PooledArray{S, R, N}},
    pa::PooledArray{T, R, N},
) where {S, T, R <: Integer, N} =
    PooledArray(RefArray(copy(pa.refs)), convert(Dict{S, R}, pa.invpool))
Base.convert(
    ::Type{PooledArray{T, R, N}},
    pa::PooledArray{T, R, N},
) where {T, R <: Integer, N} = pa
Base.convert(
    ::Type{PooledArray{S, R1}},
    pa::PooledArray{T, R2, N},
) where {S, T, R1 <: Integer, R2 <: Integer, N} = convert(PooledArray{S, R1, N}, pa)
Base.convert(
    ::Type{PooledArray{S}},
    pa::PooledArray{T, R, N},
) where {S, T, R <: Integer, N} = convert(PooledArray{S, R, N}, pa)
Base.convert(::Type{PooledArray}, pa::PooledArray{T, R, N}) where {T, R <: Integer, N} = pa

Base.convert(
    ::Type{PooledArray{S, R, N}},
    a::AbstractArray{T, N},
) where {S, T, R <: Integer, N} = PooledArray(convert(Array{S, N}, a), R)
Base.convert(
    ::Type{PooledArray{S, R}},
    a::AbstractArray{T, N},
) where {S, T, R <: Integer, N} = PooledArray(convert(Array{S, N}, a), R)
Base.convert(::Type{PooledArray{S}}, a::AbstractArray{T, N}) where {S, T, N} =
    PooledArray(convert(Array{S, N}, a))
Base.convert(::Type{PooledArray}, a::AbstractArray) = PooledArray(a)

function Base.convert(::Type{Array{S, N}}, pa::PooledArray{T, R, N}) where {S, T, R, N}
    res = Array{S}(undef, size(pa))
    for i in 1:length(pa)
        if pa.refs[i] != 0
            res[i] = pa.pool[pa.refs[i]]
        end
    end
    return res
end

Base.convert(::Type{Vector}, pv::PooledVector{T, R}) where {T, R} = convert(Array{T, 1}, pv)

Base.convert(::Type{Matrix}, pm::PooledMatrix{T, R}) where {T, R} = convert(Array{T, 2}, pm)

Base.convert(::Type{Array}, pa::PooledArray{T, R, N}) where {T, R, N} =
    convert(Array{T, N}, pa)

##############################################################################
##
## indexing
##
##############################################################################

# Scalar case
Base.@propagate_inbounds function Base.getindex(pa::PooledArray, I::Integer...)
    idx = pa.refs[I...]
    iszero(idx) && throw(UndefRefError())
    return @inbounds pa.pool[idx]
end

Base.@propagate_inbounds function Base.isassigned(pa::PooledArray, I::Int...)
    return !iszero(pa.refs[I...])
end

# Vector case
Base.@propagate_inbounds function Base.getindex(
    A::PooledArray,
    I::Union{Real, AbstractVector}...,
)
    return PooledArray(RefArray(getindex(A.refs, I...)), copy(A.invpool))
end

# Dispatch our implementation for these cases instead of Base
Base.@propagate_inbounds Base.getindex(A::PooledArray, I::AbstractVector) =
    PooledArray(RefArray(getindex(A.refs, I)), copy(A.invpool))
Base.@propagate_inbounds Base.getindex(A::PooledArray, I::AbstractArray) =
    PooledArray(RefArray(getindex(A.refs, I)), copy(A.invpool))

##############################################################################
##
## setindex!() definitions
##
##############################################################################

function getpoolidx(pa::PooledArray{T, R}, val::Any) where {T, R}
    val::T = convert(T, val)
    pool_idx = get(pa.invpool, val, zero(R))
    if pool_idx == zero(R)
        pool_idx = unsafe_pool_push!(pa, val)
    end
    return pool_idx
end

function unsafe_pool_push!(pa::PooledArray{T, R}, val) where {T, R}
    _pool_idx = length(pa.pool) + 1
    if _pool_idx > typemax(R)
        throw(ErrorException(string(
            "You're using a PooledArray with ref type $R, which can only hold $(Int(typemax(R))) values,\n",
            "and you just tried to add the $(typemax(R)+1)th reference.  Please change the ref type\n",
            "to a larger int type, or use the default ref type ($DEFAULT_POOLED_REF_TYPE).",
        )))
    end
    pool_idx = convert(R, _pool_idx)
    pa.invpool[val] = pool_idx
    push!(pa.pool, val)
    return pool_idx
end

Base.@propagate_inbounds function Base.setindex!(x::PooledArray, val, ind::Integer)
    x.refs[ind] = getpoolidx(x, val)
    return x
end

##############################################################################
##
## growing and shrinking
##
##############################################################################

function Base.push!(pv::PooledVector{S, R}, v::T) where {S, R, T}
    push!(pv.refs, getpoolidx(pv, v))
    return pv
end

function Base.append!(pv::PooledVector, items::AbstractArray)
    itemindices = eachindex(items)
    l = length(pv)
    n = length(itemindices)
    resize!(pv.refs, l + n)
    copyto!(pv, l + 1, items, first(itemindices), n)
    return pv
end

Base.pop!(pv::PooledVector) = pv.invpool[pop!(pv.refs)]

function Base.pushfirst!(pv::PooledVector{S, R}, v::T) where {S, R, T}
    pushfirst!(pv.refs, getpoolidx(pv, v))
    return pv
end

Base.popfirst!(pv::PooledVector) = pv.invpool[popfirst!(pv.refs)]

Base.empty!(pv::PooledVector) = (empty!(pv.refs); pv)

Base.deleteat!(pv::PooledVector, inds) = (deleteat!(pv.refs, inds); pv)

function _vcat!(c, a, b)
    copyto!(c, 1, a, 1, length(a))
    return copyto!(c, length(a) + 1, b, 1, length(b))
end

function Base.vcat(a::PooledArray{<:Any, <:Integer, 1}, b::AbstractArray{<:Any, 1})
    output = similar(b, promote_type(eltype(a), eltype(b)), length(b) + length(a))
    return _vcat!(output, a, b)
end

function Base.vcat(a::AbstractArray{<:Any, 1}, b::PooledArray{<:Any, <:Integer, 1})
    output = similar(a, promote_type(eltype(a), eltype(b)), length(b) + length(a))
    return _vcat!(output, a, b)
end

function Base.vcat(
    a::PooledArray{T, <:Integer, 1},
    b::PooledArray{S, <:Integer, 1},
) where {T, S}
    ap = a.invpool
    bp = b.invpool

    U = promote_type(T, S)

    poolmap = Dict{Int, Int}()
    l = length(ap)
    newlabels = Dict{U, Int}(ap)
    for (x, i) in bp
        j = if x in keys(ap)
            poolmap[i] = ap[x]
        else
            poolmap[i] = (l += 1)
        end
        newlabels[x] = j
    end
    types = [UInt8, UInt16, UInt32, UInt64]
    tidx = findfirst(t -> l < typemax(t), types)
    refT = types[tidx]
    refs2 = map(r -> convert(refT, poolmap[r]), b.refs)
    newrefs = Base.typed_vcat(refT, a.refs, refs2)
    return PooledArray(RefArray(newrefs), convert(Dict{U, refT}, newlabels))
end

function fast_sortable(y::PooledArray)
    poolranks = invperm(sortperm(y.pool))
    newpool = Dict(j => convert(eltype(y.refs), i) for (i, j) in enumerate(poolranks))
    return PooledArray(RefArray(y.refs), newpool)
end

_perm(o::F, z::V) where {F, V} = Base.Order.Perm{F, V}(o, z)

Base.Order.Perm(o::Base.Order.ForwardOrdering, y::PooledArray) = _perm(o, fast_sortable(y))

end
