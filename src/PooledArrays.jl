module PooledArrays

import DataAPI
import Future.copy!

export PooledArray, PooledVector, PooledMatrix

##############################################################################
##
## PooledArray type definition
##
##############################################################################

const DEFAULT_POOLED_REF_TYPE = UInt32
const DEFAULT_SIGNED_REF_TYPE = Int32

# This is used as a wrapper during PooledArray construction only, to distinguish
# arrays of pool indices from normal arrays
mutable struct RefArray{R}
    a::R
end

function _invert(d::Dict{K,V}) where {K,V}
    d1 = Vector{K}(undef, length(d))
    for (k, v) in d
        d1[v] = k
    end
    return d1
end

mutable struct PooledArray{T, R<:Integer, N, RA} <: AbstractArray{T, N}
    refs::RA
    pool::Vector{T}
    invpool::Dict{T,R}
    # refcount[] is 1 if only one PooledArray holds a reference to pool and invpool
    refcount::Threads.Atomic{Int}

    function PooledArray{T,R,N,RA}(rs::RefArray{RA}, invpool::Dict{T, R},
                                   pool::Vector{T}=_invert(invpool),
                                   refcount::Threads.Atomic{Int}=Threads.Atomic{Int}(1)) where {T,R,N,RA<:AbstractArray{R, N}}
        # we currently support only 1-based indexing for refs
        # TODO: change to Base.require_one_based_indexing after we drop Julia 1.0 support
        for ax in axes(rs.a)
            if first(ax) != 1
                throw(ArgumentError("offset arrays are not supported but got an array with index other than 1"))
            end
        end

        # this is a quick but incomplete consistency check
        if length(pool) != length(invpool)
            throw(ArgumentError("inconsistent pool and invpool"))
        end
        if length(rs.a) > 0
            # 0 indicates #undef
            # refs mustn't overflow pool
            minref, maxref = extrema(rs.a)
            if (minref < 0 || maxref > length(invpool))
                throw(ArgumentError("Reference array points beyond the end of the pool"))
            end
        end
        pa = new{T,R,N,RA}(rs.a, pool, invpool, refcount)
        finalizer(x -> Threads.atomic_sub!(x.refcount, 1), pa)
        return pa
    end
end
const PooledVector{T,R} = PooledArray{T,R,1}
const PooledMatrix{T,R} = PooledArray{T,R,2}

const PooledArrOrSub = Union{SubArray{T, N, <:PooledArray{T, R}},
                             PooledArray{T, R, N}} where {T, N, R}

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
@inline PooledArray(refs::RefArray{RA}, invpool::Dict{T,R}, pool::Vector{T}=_invert(invpool),
            refcount::Threads.Atomic{Int}=Threads.Atomic{Int}(1)) where {T,R,RA<:AbstractArray{R}} =
    PooledArray{T,R,ndims(RA),RA}(refs, invpool, pool, refcount)

# workaround https://github.com/JuliaLang/julia/pull/39809
_our_copy(x) = copy(x)

function _our_copy(x::SubArray{<:Any, 0})
    y = similar(x)
    y[] = x[]
    return y
end

@inline function PooledArray(d::PooledArrOrSub)
    Threads.atomic_add!(refcount(d), 1)
    return PooledArray(RefArray(_our_copy(DataAPI.refarray(d))),
                       DataAPI.invrefpool(d), DataAPI.refpool(d), refcount(d))
end

function _label(xs::AbstractArray,
                ::Type{T}=eltype(xs),
                ::Type{I}=DEFAULT_POOLED_REF_TYPE,
                start = 1,
                labels = Array{I}(undef, size(xs)),
                invpool::Dict{T,I} = Dict{T, I}(),
                pool::Vector{T} = T[],
                nlabels = 0,
               ) where {T, I<:Integer}

    @inbounds for i in start:length(xs)
        idx = i + firstindex(xs) - 1
        if !isassigned(xs, idx)
            labels[i] = zero(I)
        else
            x = xs[idx]
            lbl = get(invpool, x, zero(I))
            if lbl !== zero(I)
                labels[i] = lbl
            else
                if nlabels == typemax(I)
                    I2 = _widen(I)
                    return _label(xs, T, I2, i, convert(Vector{I2}, labels),
                                convert(Dict{T, I2}, invpool), pool, nlabels)
                end
                nlabels += 1
                labels[i] = nlabels
                invpool[x] = nlabels
                push!(pool, x)
            end
        end
    end
    labels, invpool, pool
end

_widen(::Type{UInt8}) = UInt16
_widen(::Type{UInt16}) = UInt32
_widen(::Type{UInt32}) = UInt64
_widen(::Type{Int8}) = Int16
_widen(::Type{Int16}) = Int32
_widen(::Type{Int32}) = Int64

# Constructor from array, invpool, and ref type

"""
    PooledArray(array, [reftype]; signed=false, compress=false)

Freshly allocate `PooledArray` using the given array as a source where each
element will be referenced as an integer of the given type.

If `reftype` is not specified then `PooledArray` constructor is not type stable.
In this case Boolean keyword arguments `signed` and `compress`
determine the type of integer references. By default (`signed=false`), unsigned integers
are used, as they have a greater range.
However, the Arrow standard at https://arrow.apache.org/, as implemented in
the Arrow package, requires signed integer types, which are provided when `signed=true`.
When `compress=false`, `reftype` is a 32-bits type (`UInt32` for unsigned, `Int32` for signed);
when `compress=true`, `reftype` is chosen to be the smallest integer type that is
large enough to hold all unique values in `array`.

Note that if you hold mutable objects in `PooledArray` it is not allowed to modify them
after they are stored in it.

In order to improve performance of `getindex` and `copyto!` operations `PooledArray`s
may share pools. This sharing is automatically undone by copying a shared pool before
adding new values to it.

It is not safe to assign values that are not already present in a `PooledArray`'s pool
from one thread while either reading or writing to the same array from another thread
(even if pools are not shared). However, reading and writing from different threads is safe
if all values already exist in the pool.
"""
PooledArray

@inline function PooledArray{T}(d::AbstractArray, r::Type{R}) where {T,R<:Integer}
    refs, invpool, pool = _label(d, T, R)

    if length(invpool) > typemax(R)
        throw(ArgumentError("Cannot construct a PooledArray with type $R with a pool of size $(length(pool))"))
    end

    # Assertions are needed since _label is not type stable
    return PooledArray(RefArray(refs::Array{R, ndims(d)}), invpool::Dict{T,R}, pool)
end

@inline function PooledArray{T}(d::AbstractArray; signed::Bool=false, compress::Bool=false) where {T}
    R = signed ? (compress ? Int8 : DEFAULT_SIGNED_REF_TYPE) : (compress ? UInt8 : DEFAULT_POOLED_REF_TYPE)
    refs, invpool, pool = _label(d, T, R)
    return PooledArray(RefArray(refs), invpool, pool)
end

@inline PooledArray(d::AbstractArray{T}, r::Type) where {T} = PooledArray{T}(d, r)
@inline PooledArray(d::AbstractArray{T}; signed::Bool=false, compress::Bool=false) where {T} =
    PooledArray{T}(d, signed=signed, compress=compress)

# Construct an empty PooledVector of a specific type
@inline PooledArray(t::Type) = PooledArray(Array(t,0))
@inline PooledArray(t::Type, r::Type) = PooledArray(Array(t,0), r)

##############################################################################
##
## Basic interface functions
##
##############################################################################

DataAPI.refarray(pa::PooledArray) = pa.refs
DataAPI.refvalue(pa::PooledArray, i::Integer) = pa.pool[i]
DataAPI.refpool(pa::PooledArray) = pa.pool
DataAPI.invrefpool(pa::PooledArray) = pa.invpool
refcount(pa::PooledArray) = pa.refcount

DataAPI.refarray(pav::SubArray{<:Any, <:Any, <:PooledArray}) = view(parent(pav).refs, pav.indices...)
DataAPI.refvalue(pav::SubArray{<:Any, <:Any, <:PooledArray}, i::Integer) = parent(pav).pool[i]
DataAPI.refpool(pav::SubArray{<:Any, <:Any, <:PooledArray}) = parent(pav).pool
DataAPI.invrefpool(pav::SubArray{<:Any, <:Any, <:PooledArray}) = parent(pav).invpool
refcount(pav::SubArray{<:Any, <:Any, <:PooledArray}) = parent(pav).refcount

Base.size(pa::PooledArray) = size(pa.refs)
Base.length(pa::PooledArray) = length(pa.refs)
Base.lastindex(pa::PooledArray) = lastindex(pa.refs)

Base.copy(pa::PooledArrOrSub) = PooledArray(pa)

# here we do not allow dest to be SubArray as copy! is intended to replace whole arrays
# slow path will be used for SubArray
function copy!(dest::PooledArray{T, R, N},
               src::PooledArrOrSub{T, N, R}) where {T, N, R}
    copy!(dest.refs, DataAPI.refarray(src))
    src_refcount = refcount(src)

    if dest.pool !== DataAPI.refpool(src)
        Threads.atomic_sub!(dest.refcount, 1)
        Threads.atomic_add!(src_refcount, 1)
        dest.pool = DataAPI.refpool(src)
        dest.invpool = DataAPI.invrefpool(src)
        dest.refcount = src_refcount
    else
        @assert dest.invpool === DataAPI.invrefpool(src)
        @assert dest.refcount === src_refcount
    end
    return dest
end

# this is needed as Julia Base uses a special path for this case we want to avoid
Base.copyto!(dest::PooledArrOrSub{T, N, R}, src::PooledArrOrSub{T, N, R}) where {T, N, R} =
    copyto!(dest, 1, src, 1, length(src))

function Base.copyto!(dest::PooledArrOrSub{T, N, R}, doffs::Union{Signed, Unsigned},
                      src::PooledArrOrSub{T, N, R}, soffs::Union{Signed, Unsigned},
                      n::Union{Signed, Unsigned}) where {T, N, R}
    n == 0 && return dest
    n > 0 || Base._throw_argerror()
    if soffs < 1 || doffs < 1 || soffs + n - 1 > length(src) || doffs + n - 1 > length(dest)
        throw(BoundsError())
    end

    dest_pa = dest isa PooledArray ? dest : parent(dest)
    src_refcount = refcount(src)

    # if dest_pa.pool is empty we can safely replace it as we are sure it holds
    # no information; having this path is useful because then we can efficiently
    # `copyto!` into a fresh `PooledArray` created using the `similar` function
    if DataAPI.refpool(dest) === DataAPI.refpool(src)
        @assert DataAPI.invrefpool(dest) === DataAPI.invrefpool(src)
        @assert refcount(dest) === refcount(src)
        copyto!(DataAPI.refarray(dest), doffs, DataAPI.refarray(src), soffs, n)
    elseif length(dest_pa.pool) == 0
        @assert length(dest_pa.invpool) == 0
        Threads.atomic_add!(src_refcount, 1)
        dest_pa.pool = DataAPI.refpool(src)
        dest_pa.invpool = DataAPI.invrefpool(src)
        Threads.atomic_sub!(dest_pa.refcount, 1)
        dest_pa.refcount = src_refcount
        copyto!(DataAPI.refarray(dest), doffs, DataAPI.refarray(src), soffs, n)
    else
        @inbounds for i in 0:n-1
            dest[doffs+i] = src[soffs+i]
        end
    end
    return dest
end

function Base.resize!(pa::PooledArray{T,R,1}, n::Integer) where {T,R}
    oldn = length(pa.refs)
    resize!(pa.refs, n)
    pa.refs[oldn+1:n] .= zero(R)
    return pa
end

function Base.reverse(x::PooledArray)
    Threads.atomic_add!(x.refcount, 1)
    PooledArray(RefArray(reverse(x.refs)), x.invpool, x.pool, x.refcount)
end

function Base.permute!(x::PooledArray, p::AbstractVector{T}) where T<:Integer
    permute!(x.refs, p)
    return x
end

function Base.invpermute!(x::PooledArray, p::AbstractVector{T}) where T<:Integer
    invpermute!(x.refs, p)
    return x
end

Base.similar(pa::PooledArray{T,R}, S::Type, dims::Dims) where {T,R} =
    PooledArray(RefArray(zeros(R, dims)), Dict{S,R}())

Base.findall(pdv::PooledVector{Bool}) = findall(convert(Vector{Bool}, pdv))

##############################################################################
##
## map
## Calls `f` only once per pool entry.
##
##############################################################################

"""
    map(f, x::PooledArray; pure::Bool=false)

Transform `PooledArray` `x` by applying `f` to each element.

If `pure=true` then `f` is applied to each element of pool of `x`
exactly once (even if some elements in pool are not present it `x`).
This will typically be much faster when the proportion of unique values
in `x` is small.

If `pure=false`, the returned array will use the same reference type
as `x`, or `Int` if the number of unique values in the result is too large
to fit in that type.
"""
function Base.map(f, x::PooledArray{<:Any, R, N, RA}; pure::Bool=false)::Union{PooledArray{<:Any, R, N, RA},
                                                                               PooledArray{<:Any, Int, N,
                                                                                           typeof(similar(x.refs, Int, ntuple(i -> 0, ndims(x.refs))))}} where {R, N, RA}
    pure && return _map_pure(f, x)
    length(x) == 0 && return PooledArray([f(v) for v in x])
    v1 = f(x[1])
    invpool = Dict(v1 => one(eltype(x.refs)))
    pool = [v1]
    labels = similar(x.refs)
    labels[1] = 1
    nlabels = 1
    return _map_notpure(f, x, 2, invpool, pool, labels, nlabels)
end

function _map_notpure(f, xs::PooledArray, start,
                      invpool::Dict{T,I}, pool::Vector{T},
                      labels::AbstractArray{I}, nlabels::Int) where {T, I<:Integer}
    for i in start:length(xs)
        vi = f(xs[i])
        lbl = get(invpool, vi, zero(I))
        if lbl != zero(I)
            labels[i] = lbl
        else
            if nlabels == typemax(I) || !(vi isa T)
                I2 = nlabels == typemax(I) ? Int : I
                T2 = vi isa T ? T : Base.promote_typejoin(T, typeof(vi))
                nlabels += 1
                invpool2 = convert(Dict{T2, I2}, invpool)
                invpool2[vi] = nlabels
                pool2 = convert(Vector{T2}, pool)
                push!(pool2, vi)
                labels2 = convert(AbstractArray{I2}, labels)
                labels2[i] = nlabels
                return _map_notpure(f, xs, i + 1, invpool2, pool2,
                                    labels2, nlabels)
            end
            nlabels += 1
            labels[i] = nlabels
            invpool[vi] = nlabels
            push!(pool, vi)
        end
    end
    return PooledArray(RefArray(labels), invpool, pool)
end

function _map_pure(f, x::PooledArray)
    ks = collect(keys(x.invpool))
    vs = collect(values(x.invpool))
    ks1 = map(f, ks)
    uks = Set(ks1)
    if length(uks) < length(ks1)
        # this means some keys have repeated
        newinvpool = Dict{eltype(ks1), eltype(vs)}()
        translate = Dict{eltype(vs), eltype(vs)}()
        i = 1
        for (k, k1) in zip(ks, ks1)
            if haskey(newinvpool, k1)
                translate[x.invpool[k]] = newinvpool[k1]
            else
                newinvpool[k1] = i
                translate[x.invpool[k]] = i
                i+=1
            end
        end
        refarray = map(x->translate[x], x.refs)
    else
        newinvpool = Dict(zip(ks1, vs))
        refarray = copy(x.refs)
    end
    return PooledArray(RefArray(refarray), newinvpool)
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
                       poolperm = sortperm(pa.pool, alg=alg, order=_ord))

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

function Base.convert(::Type{PooledArray{S,R1,N}}, pa::PooledArray{T,R2,N}) where {S,T,R1<:Integer,R2<:Integer,N}
    invpool_conv = convert(Dict{S,R1}, pa.invpool)
    @assert invpool_conv !== pa.invpool

    if R1 === R2
        refs_conv = pa.refs
    else
        refs_conv = convert(Array{R1,N}, pa.refs)
        @assert refs_conv !== pa.refs
    end

    return PooledArray(RefArray(refs_conv), invpool_conv)
end

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
            res[i] = pa.pool[pa.refs[i]]
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

# We need separate functions due to dispatch ambiguities

Base.@propagate_inbounds function Base.getindex(A::PooledArray, I::Int)
    idx = DataAPI.refarray(A)[I]
    iszero(idx) && throw(UndefRefError())
    return @inbounds DataAPI.refpool(A)[idx]
end

# we handle fast only the case when the first index is an abstract vector
# this is to make sure other indexing synraxes use standard dispatch from Base
# the reason is that creation of DataAPI.refarray(A) is unfortunately slow
Base.@propagate_inbounds function Base.getindex(A::PooledArrOrSub,
                                                I1::AbstractVector,
                                                I2::Union{Real, AbstractVector}...)
    # make sure we do not increase A.refcount in case creation of newrefs fails
    newrefs = DataAPI.refarray(A)[I1, I2...]
    @assert newrefs isa AbstractArray
    Threads.atomic_add!(refcount(A), 1)
    return PooledArray(RefArray(newrefs), DataAPI.invrefpool(A), DataAPI.refpool(A), refcount(A))
end

Base.@propagate_inbounds function Base.isassigned(pa::PooledArrOrSub, I::Int...)
    !iszero(DataAPI.refarray(pa)[I...])
end

##############################################################################
##
## setindex!() definitions
##
##############################################################################

function getpoolidx(pa::PooledArray{T,R}, val::Any) where {T,R}
    val::T = convert(T,val)
    pool_idx = get(pa.invpool, val, zero(R))
    if pool_idx == zero(R)
        pool_idx = unsafe_pool_push!(pa, val)
    end
    return pool_idx
end

function unsafe_pool_push!(pa::PooledArray{T,R}, val) where {T,R}
    # Warning - unsafe_pool_push! may not be used in any multithreaded context
    _pool_idx = length(pa.pool) + 1
    if _pool_idx > typemax(R)
        throw(ErrorException(string(
            "You're using a PooledArray with ref type $R, which can only hold $(Int(typemax(R))) values,\n",
            "and you just tried to add the $(typemax(R)+1)th reference.  Please change the ref type\n",
            "to a larger int type, or use the default ref type ($DEFAULT_POOLED_REF_TYPE)."
           )))
    end
    pool_idx = convert(R, _pool_idx)
    if pa.refcount[] > 1
        pa.invpool = copy(pa.invpool)
        pa.pool = copy(pa.pool)
        Threads.atomic_sub!(pa.refcount, 1)
        pa.refcount = Threads.Atomic{Int}(1)
    end
    pa.invpool[val] = pool_idx
    push!(pa.pool, val)
    pool_idx
end

# assume PooledArray is only used with Arrays as this is what _label does
# this simplifies code below
Base.IndexStyle(::Type{<:PooledArray}) = IndexLinear()

Base.@propagate_inbounds function Base.setindex!(x::PooledArray, val, ind::Int)
    x.refs[ind] = getpoolidx(x, val)
    return x
end

##############################################################################
##
## growing and shrinking
##
##############################################################################

function Base.push!(pv::PooledVector, v) # this function is not thread safe
    push!(pv.refs, getpoolidx(pv, v))
    return pv
end

function Base.insert!(pv::PooledVector, i::Integer, v) # this function is not thread safe
    i isa Bool && throw(ArgumentError("invalid index: $i of type Bool"))
    if !(1 <= i <= length(pv.refs) + 1)
        throw(BoundsError("attempt to insert to a vector with length $(length(pv)) at index $i"))
    end
    insert!(pv.refs, i, getpoolidx(pv, v))
    return pv
end

function Base.append!(pv::PooledVector, items::AbstractArray)
    itemindices = eachindex(items)
    l = length(pv)
    n = length(itemindices)
    resize!(pv.refs, l+n)
    copyto!(pv, l+1, items, first(itemindices), n)
    return pv
end

Base.pop!(pv::PooledVector) = pv.pool[pop!(pv.refs)]

function Base.pushfirst!(pv::PooledVector{S,R}, v::T) where {S,R,T}
    pushfirst!(pv.refs, getpoolidx(pv, v))
    return pv
end

Base.popfirst!(pv::PooledVector) = pv.pool[popfirst!(pv.refs)]

Base.empty!(pv::PooledVector) = (empty!(pv.refs); pv)

Base.deleteat!(pv::PooledVector, inds) = (deleteat!(pv.refs, inds); pv)

function _vcat!(c, a, b)
    copyto!(c, 1, a, 1, length(a))
    return copyto!(c, length(a)+1, b, 1, length(b))
end

function Base.vcat(a::PooledArray{<:Any, <:Integer, 1}, b::AbstractArray{<:Any, 1})
    output = similar(b, promote_type(eltype(a), eltype(b)), length(b) + length(a))
    return _vcat!(output, a, b)
end

function Base.vcat(a::AbstractArray{<:Any, 1}, b::PooledArray{<:Any, <:Integer, 1})
    output = similar(a, promote_type(eltype(a), eltype(b)), length(b) + length(a))
    return _vcat!(output, a, b)
end

function Base.vcat(a::PooledArray{T, <:Integer, 1}, b::PooledArray{S, <:Integer, 1}) where {T, S}
    ap = a.invpool
    bp = b.invpool

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

fast_sortable(y::PooledArray) = _fast_sortable(y)
fast_sortable(y::PooledArray{T}) where {T<:Integer} = isbitstype(T) ? y : _fast_sortable(y)

function _fast_sortable(y::PooledArray)
    poolranks = invperm(sortperm(y.pool))
    newpool = Dict(j=>convert(eltype(y.refs), i) for (i,j) in enumerate(poolranks))
    PooledArray(RefArray(y.refs), newpool)
end

_perm(o::F, z::V) where {F, V} = Base.Order.Perm{F, V}(o, z)

Base.Order.Perm(o::Base.Order.ForwardOrdering, y::PooledArray) = _perm(o, fast_sortable(y))

function Base.repeat(x::PooledArray, m::Integer...)
    Threads.atomic_add!(x.refcount, 1)
    PooledArray(RefArray(repeat(x.refs, m...)), x.invpool, x.pool, x.refcount)
end

function Base.repeat(x::PooledArray; inner = nothing, outer = nothing)
    Threads.atomic_add!(x.refcount, 1)
    PooledArray(RefArray(repeat(x.refs; inner = inner, outer = outer)),
                                x.invpool, x.pool, x.refcount)
end

end
