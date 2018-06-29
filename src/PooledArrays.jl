__precompile__()

module PooledArrays

export PooledArray

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

function _invert(d::Dict{K,V}) where {K,V}
    d1 = Dict{V,K}()
    for (k, v) in d
        d1[v] = k
    end
    d1
end

mutable struct PooledArray{T, R<:Integer, N, RA} <: AbstractArray{T, N}
    refs::RA
    pool::Dict{T,R}
    revpool::Dict{R,T}

    function PooledArray(rs::RefArray{RA}, p::Dict{T, R}, revpool=_invert(p)) where {T,R,N,RA<:AbstractArray{R, N}}
        # refs mustn't overflow pool
        if length(rs.a) > 0 && maximum(rs.a) > length(p)
            throw(ArgumentError("Reference array points beyond the end of the pool"))
        end
        new{T,R,N,RA}(rs.a,p,revpool)
    end
end
const PooledVector{T,R} = PooledArray{T,R,1}
const PooledMatrix{T,R} = PooledArray{T,R,2}

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
function PooledArray(refs::RefArray{R}, pool::Dict{T,R}) where {T,R}
    PooledArray{T,eltype(R),ndims(R),R}(refs, pool)
end

# A no-op constructor
PooledArray(d::PooledArray) = d

function _label(xs::AbstractArray{T},
                ::Type{I}=UInt8,
                start = 1,
                labels = Array{I}(undef, size(xs)),
                pool::Dict{T,I} = Dict{T, I}(),
                nlabels = 0,
               ) where {T, I<:Integer}

    @inbounds for i in start:length(xs)
        x = xs[i]
        lbl = get(pool, x, zero(I))
        if lbl !== zero(I)
            labels[i] = lbl
        else
            if nlabels == typemax(I)
                I2 = _widen(I)
                return _label(xs, I2, i, convert(Vector{I2}, labels),
                              convert(Dict{T, I2}, pool), nlabels)
            end
            nlabels += 1
            labels[i] = convert(I, nlabels)
            pool[x] = convert(I, nlabels)
        end
    end
    labels, pool
end

_widen(::Type{UInt8}) = UInt16
_widen(::Type{UInt16}) = UInt32
_widen(::Type{UInt32}) = UInt64

# Constructor from array, pool, and ref type

"""
    PooledArray(array, [reftype])

Convert the given array to a PooledArray where each element will be referenced
as an integer of the given type. If no `reftype` is specified one is chosen
automatically based on the number of unique elements.
"""
PooledArray

function PooledArray{T}(d::AbstractArray, r::Type{R}) where {T,R<:Integer}
    refs, pool = _label(d)

    if length(pool) > typemax(R)
        throw(ArgumentError("Cannot construct a PooledArray with type $R with a pool of size $(length(pool))"))
    end

    refs1 = convert(Vector{R}, refs)
    pool1 = convert(Dict{T,R}, pool)
    PooledArray(RefArray(refs1), pool1)
end

function PooledArray{T}(d::AbstractArray) where T
    refs, pool = _label(d)
    PooledArray(RefArray(refs), pool)
end

PooledArray(d::AbstractArray{T}, r::Type{R}) where {T,R<:Integer} = PooledArray{T}(d, r)
PooledArray(d::AbstractArray{T}) where {T} = PooledArray{T}(d)

# Construct an empty PooledVector of a specific type
PooledArray(t::Type) = PooledArray(Array(t,0))
PooledArray(t::Type, r::Type{R}) where {R<:Integer} = PooledArray(Array(t,0), r)

##############################################################################
##
## Basic interface functions
##
##############################################################################

Base.size(pa::PooledArray) = size(pa.refs)
Base.length(pa::PooledArray) = length(pa.refs)
Base.endof(pa::PooledArray) = endof(pa.refs)

Base.copy(pa::PooledArray) = PooledArray(RefArray(copy(pa.refs)), copy(pa.pool))
# TODO: Implement copy_to()

function Base.resize!(pa::PooledArray{T,R,1}, n::Integer) where {T,R}
    oldn = length(pa.refs)
    resize!(pa.refs, n)
    pa.refs[oldn+1:n] = zero(R)
    pa
end

Base.reverse(x::PooledArray) = PooledArray(RefArray(reverse(x.refs)), x.pool)

function Base.permute!!(x::PooledArray, p::AbstractVector{T}) where T<:Integer
    Base.permute!!(x.refs, p)
    x
end

function Base.invpermute!!(x::PooledArray, p::AbstractVector{T}) where T<:Integer
    Base.invpermute!!(x.refs, p)
    x
end

function Base.similar(pa::PooledArray{T,R}, S::Type, dims::Dims) where {T,R}
    PooledArray(RefArray(zeros(R, dims)), Dict{S,R}())
end

Base.find(pdv::PooledVector{Bool}) = findall(convert(Vector{Bool}, pdv, false))

##############################################################################
##
## map
## Calls `f` only once per pool entry.
##
##############################################################################

function Base.map(f, x::PooledArray{T,R}) where {T,R<:Integer}
    ks = collect(keys(x.pool))
    vs = collect(values(x.pool))
    ks1 = map(f, ks)
    uks = Set(ks1)
    if length(uks) < length(ks1)
        # this means some keys have repeated
        newpool = Dict{eltype(ks1), eltype(vs)}()
        translate = Dict{eltype(vs), eltype(vs)}()
        i = 1
        for (k, k1) in zip(ks, ks1)
            if haskey(newpool, k1)
                translate[x.pool[k]] = newpool[k1]
            else
                newpool[k1] = i
                translate[x.pool[k]] = i
                i+=1
            end
        end
        refarray = map(x->translate[x], x.refs)
    else
        newpool = Dict(zip(map(f, ks), vs))
        refarray = copy(x.refs)
    end
    PooledArray(RefArray(refarray), newpool)
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
    # counts = x.pool
    counts = fill(0, ngroups + 1)
    @inbounds for i = 1:n
        counts[x[i] + 1] += 1
    end
    counts[2:end] = counts[perm.+1]

    # mark the start of each contiguous group of like-indexed data
    where = fill(1, ngroups + 1)
    @inbounds for i = 2:ngroups+1
        where[i] = where[i - 1] + counts[i - 1]
    end

    # this is our indexer
    result = fill(0, n)
    iperm = invperm(perm)

    @inbounds for i = 1:n
        label = iperm[x[i]] + 1
        result[where[label]] = i
        where[label] += 1
    end
    result, where, counts
end

function Base.sortperm(pa::PooledArray; alg::Base.Sort.Algorithm=Base.Sort.DEFAULT_UNSTABLE,
                       lt::Function=isless, by::Function=identity,
                       rev::Bool=false, order=Base.Sort.Forward,
                       _ord = Base.ord(lt, by, rev, order),
                       poolperm = sortperm([pa.revpool[i] for i=1:length(pa.pool)], alg=alg, order=_ord))

    groupsort_indexer(pa.refs, length(pa.pool), poolperm)[1]
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

Base.convert(::Type{PooledArray{S,R1,N}}, pa::PooledArray{T,R2,N}) where {S,T,R1<:Integer,R2<:Integer,N} =
    PooledArray(RefArray(convert(Array{R1,N}, pa.refs)), convert(Dict{S,R1}, pa.pool))
Base.convert(::Type{PooledArray{S,R,N}}, pa::PooledArray{T,R,N}) where {S,T,R<:Integer,N} =
    PooledArray(RefArray(copy(pa.refs)), convert(Dict{S,R}, pa.pool))
Base.convert(::Type{PooledArray{T,R,N}}, pa::PooledArray{T,R,N}) where {T,R<:Integer,N} = pa
Base.convert(::Type{PooledArray{S,R1}}, pa::PooledArray{T,R2,N}) where {S,T,R1<:Integer,R2<:Integer,N} =
    convert(PooledArray{S,R1,N}, pa)
Base.convert(::Type{PooledArray{S}}, pa::PooledArray{T,R,N}) where {S,T,R<:Integer,N} =
    convert(PooledArray{S,R,N}, pa)
Base.convert(::Type{PooledArray}, pa::PooledArray{T,R,N}) where {T,R<:Integer,N} = pa

Base.convert(::Type{PooledArray{S,R,N}}, a::AbstractArray{T,N}) where {S,T,R<:Integer,N} =
    PooledArray(convert(Array{S,N}, a), R)
Base.convert(::Type{PooledArray{S,R}}, a::AbstractArray{T,N}) where {S,T,R<:Integer,N} =
    PooledArray(convert(Array{S,N}, a), R)
Base.convert(::Type{PooledArray{S}}, a::AbstractArray{T,N}) where {S,T,N} =
    PooledArray(convert(Array{S,N}, a))
Base.convert(::Type{PooledArray}, a::AbstractArray) =
    PooledArray(a)

function Base.convert(::Type{Array{S, N}}, pa::PooledArray{T, R, N}) where {S, T, R, N}
    res = Array{S}(undef, size(pa))
    for i in 1:length(pa)
        if pa.refs[i] != 0
            res[i] = pa.revpool[pa.refs[i]]
        end
    end
    return res
end

Base.convert(::Type{Vector}, pv::PooledVector{T, R}) where {T, R} = convert(Array{T, 1}, pv)

Base.convert(::Type{Matrix}, pm::PooledMatrix{T, R}) where {T, R} = convert(Array{T, 2}, pm)

Base.convert(::Type{Array}, pa::PooledArray{T, R, N}) where {T, R, N} = convert(Array{T, N}, pa)

##############################################################################
##
## indexing
##
##############################################################################

# Scalar case
function Base.getindex(pa::PooledArray, I::Integer...)
    idx = pa.refs[I...]
    iszero(idx) && throw(UndefRefError())
    return pa.revpool[idx]
end

function Base.isassigned(pa::PooledArray, I::Int...)
    !iszero(pa.refs[I...])
end

# Vector case
function Base.getindex(A::PooledArray, I::Union{Real,AbstractVector}...)
    PooledArray(RefArray(getindex(A.refs, I...)), copy(A.pool))
end

# Dispatch our implementation for these cases instead of Base
Base.getindex(A::PooledArray, I::AbstractVector) =
    PooledArray(RefArray(getindex(A.refs, I)), copy(A.pool))
Base.getindex(A::PooledArray, I::AbstractArray) =
    PooledArray(RefArray(getindex(A.refs, I)), copy(A.pool))

##############################################################################
##
## setindex!() definitions
##
##############################################################################

function getpoolidx(pa::PooledArray{T,R}, val::Any) where {T,R}
    val::T = convert(T,val)
    pool_idx = get(pa.pool, val, zero(R))
    if pool_idx == zero(R)
        pool_idx = unsafe_pool_push!(pa, val)
    end
    return pool_idx
end

function unsafe_pool_push!(pa::PooledArray{T,R}, val) where {T,R}
    _pool_idx = length(pa.pool)+1
    if _pool_idx > typemax(R)
        throw(ErrorException(string(
            "You're using a PooledArray with ref type $R, which can only hold $(Int(typemax(R))) values,\n",
            "and you just tried to add the $(typemax(R)+1)th reference.  Please change the ref type\n",
            "to a larger int type, or use the default ref type ($DEFAULT_POOLED_REF_TYPE)."
           )))
    end
    pool_idx = convert(R, _pool_idx)
    pa.pool[val] = pool_idx
    pa.revpool[pool_idx] = val
    pool_idx
end

function Base.setindex!(x::PooledArray, val, ind::Integer)
    x.refs[ind] = getpoolidx(x, val)
    return x
end

##############################################################################
##
## growing and shrinking
##
##############################################################################

function Base.push!(pv::PooledVector{S,R}, v::T) where {S,R,T}
    v = convert(S,v)
    push!(pv.refs, getpoolidx(pv, v))
    return v
end

Base.pop!(pv::PooledVector) = pv.pool[pop!(pv.refs)]

function Base.unshift!(pv::PooledVector{S,R}, v::T) where {S,R,T}
    v = convert(S,v)
    pushfirst!(pv.refs, getpoolidx(pv, v))
    return v
end

Base.shift!(pv::PooledVector) = pv.pool[popfirst!(pv.refs)]

Base.empty!(pv::PooledVector) = (empty!(pv.refs); pv)

function _vcat!(c,a,b)
    copy!(c, 1, a, 1, length(a))
    copy!(c, length(a)+1, b, 1, length(b))
end


function Base.vcat(a::PooledArray{<:Any, <:Integer, 1}, b::AbstractArray{<:Any, 1})
    output = similar(b, promote_type(eltype(a), eltype(b)), length(b) + length(a))
    _vcat!(output, a, b)
end

function Base.vcat(a::AbstractArray{<:Any, 1}, b::PooledArray{<:Any, <:Integer, 1})
    output = similar(a, promote_type(eltype(a), eltype(b)), length(b) + length(a))
    _vcat!(output, a, b)
end

function Base.vcat(a::PooledArray{T, <:Integer, 1}, b::PooledArray{S, <:Integer, 1}) where {T, S}
    ap = a.pool
    bp = b.pool

    U = promote_type(T,S)

    poolmap = Dict{Int, Int}()
    l = length(ap)
    newlabels = Dict{U, Int}(ap)
    for (x, i) in bp
        j = if x in keys(ap)
            poolmap[i] = ap[x]
        else
            poolmap[i] = (l+=1)
        end
        newlabels[x] = j
    end
    types = [UInt8, UInt16, UInt32, UInt64]
    tidx = findfirst(t->l < typemax(t), types)
    refT = types[tidx]
    refs2 = map(r->convert(refT, poolmap[r]), b.refs)
    newrefs = Base.typed_vcat(refT, a.refs, refs2)
    return PooledArray(RefArray(newrefs), convert(Dict{U, refT}, newlabels))
end

end
