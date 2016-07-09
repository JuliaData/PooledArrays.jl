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
type RefArray{R<:Integer,N}
    a::Array{R,N}
end

immutable PooledArray{T, R<:Integer, N} <: AbstractArray{T, N}
    refs::Array{R, N}
    pool::Vector{T}

    function PooledArray(rs::RefArray{R, N}, p::Vector{T})
        # refs mustn't overflow pool
        if length(rs.a) > 0 && maximum(rs.a) > prod(size(p))
            throw(ArgumentError("Reference array points beyond the end of the pool"))
        end
        new(rs.a,p)
    end
end
typealias PooledVector{T,R} PooledArray{T,R,1}
typealias PooledMatrix{T,R} PooledArray{T,R,2}

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
function PooledArray{T,R<:Integer,N}(refs::RefArray{R, N}, pool::Vector{T})
    PooledArray{T,R,N}(refs, pool)
end

# A no-op constructor
PooledArray(d::PooledArray) = d

# Constructor from array, pool, and ref type
function PooledArray{T,R<:Integer}(d::AbstractArray, pool::Vector{T},
                                   r::Type{R} = DEFAULT_POOLED_REF_TYPE)
    if length(pool) > typemax(R)
        throw(ArgumentError("Cannot construct a PooledVector with type $R with a pool of size $(length(pool))"))
    end

    newrefs = Array(R, size(d))
    poolref = Dict{T, R}()

    # loop through once to fill the poolref dict
    for i = 1:length(pool)
        poolref[pool[i]] = i
    end

    # fill in newrefs
    for i = 1:length(d)
        newrefs[i] = get(poolref, d[i], 0)
    end
    return PooledArray(RefArray(newrefs), pool)
end

# Constructor from array and ref type
function (::Type{PooledArray{T}}){T,R<:Integer}(d::AbstractArray,
                                                r::Type{R} = DEFAULT_POOLED_REF_TYPE)
    pool = convert(Vector{T}, unique(d))
    if method_exists(isless, (T, T))
        sort!(pool)
    end
    PooledArray(d, pool, r)
end

function PooledArray{T,R<:Integer}(d::AbstractArray{T},
                                   r::Type{R} = DEFAULT_POOLED_REF_TYPE)
    PooledArray{T}(d, r)
end

# Construct an empty PooledVector of a specific type
PooledArray(t::Type) = PooledArray(Array(t,0))
PooledArray{R<:Integer}(t::Type, r::Type{R}) = PooledArray(Array(t,0), r)

##############################################################################
##
## Basic size properties
##
##############################################################################

Base.size(pa::PooledArray) = size(pa.refs)
Base.length(pa::PooledArray) = length(pa.refs)
Base.endof(pa::PooledArray) = endof(pa.refs)

##############################################################################
##
## Copying
##
##############################################################################

Base.copy(pa::PooledArray) = PooledArray(RefArray(copy(pa.refs)), copy(pa.pool))
# TODO: Implement copy_to()

function Base.resize!{T,R}(pa::PooledArray{T,R,1}, n::Integer)
    oldn = length(pa.refs)
    resize!(pa.refs, n)
    pa.refs[oldn+1:n] = zero(R)
    pa
end

##############################################################################
##
## PooledArray utilities
##
##############################################################################

function compact{T,R<:Integer,N}(d::PooledArray{T,R,N})
    sz = length(d.pool)

    REFTYPE = sz <= typemax(UInt8)  ? UInt8 :
              sz <= typemax(UInt16) ? UInt16 :
              sz <= typemax(UInt32) ? UInt32 :
                                      UInt64

    if REFTYPE == R
        return d
    end

    newrefs = convert(Array{REFTYPE, N}, d.refs)
    PooledArray(RefArray(newrefs), d.pool)
end

Base.reverse(x::PooledArray) = PooledArray(RefArray(reverse(x.refs)), x.pool)

function Base.permute!!{T<:Integer}(x::PooledArray, p::AbstractVector{T})
    Base.permute!!(x.refs, p)
    x
end

function Base.ipermute!!{T<:Integer}(x::PooledArray, p::AbstractVector{T})
    Base.ipermute!!(x.refs, p)
    x
end

##############################################################################
##
## similar()
##
##############################################################################

function Base.similar{T,R}(pa::PooledArray{T,R}, S::Type, dims::Dims)
    PooledArray(RefArray(zeros(R, dims)), S[])
end

##############################################################################
##
## find()
##
##############################################################################

Base.find(pdv::PooledVector{Bool}) = find(convert(Vector{Bool}, pdv, false))

##############################################################################
##
## setindex!() definitions
##
##############################################################################

function getpoolidx{T,R}(pa::PooledArray{T,R}, val::Any)
    val::T = convert(T,val)
    pool_idx = findfirst(pa.pool, val)
    if pool_idx <= 0
        push!(pa.pool, val)
        pool_idx = length(pa.pool)
        if pool_idx > typemax(R)
            throw(ErrorException(
                "You're using a PooledArray with ref type $R, which can only hold $(int(typemax(R))) values,\n",
                "and you just tried to add the $(typemax(R)+1)th reference.  Please change the ref type\n",
                "to a larger int type, or use the default ref type ($DEFAULT_POOLED_REF_TYPE)."
            ))
        end
    end
    return pool_idx
end

##############################################################################
##
## Replacement operations
##
##############################################################################

function replace!{R, S, T}(x::PooledArray{R}, fromval::S, toval::T)
    # throw error if fromval isn't in the pool
    fromidx = findfirst(x.pool, fromval)
    if fromidx == 0
        throw(ErrorException("can't replace a value not in the pool in a PooledArray!"))
    end

    # if toval is in the pool too, use that and remove fromval from the pool
    toidx = findfirst(x.pool, toval)
    if toidx != 0
        x.refs[x.refs .== fromidx] = toidx
        #x.pool[fromidx] = None    TODO: what to do here??
    else
        # otherwise, toval is new, swap it in
        x.pool[fromidx] = toval
    end

    return toval
end

##############################################################################
##
## Sorting can use the pool to speed things up
##
##############################################################################
#=
function Base.sortperm(pa::PooledArray; alg::Base.Sort.Algorithm=Base.Sort.DEFAULT_UNSTABLE,
                       lt::Function=isless, by::Function=identity,
                       rev::Bool=false, order=Base.Sort.Forward)
    order = Base.ord(lt, by, rev, order)

    # TODO handle custom ordering efficiently
    if !isa(order, Base.Order.ForwardOrdering) && !isa(order, Base.Order.ReverseOrdering)
        return sort!(collect(1:length(pda)), alg, Base.Order.Perm(order,pda))
    end

    # TODO handle non-sorted keys without copying
    perm = issorted(pda.pool) ? groupsort_indexer(pda, true)[1] : sortperm(reorder(pda))
    isa(order, Base.Order.ReverseOrdering) && reverse!(perm)
    perm
end

Base.sort(pda::PooledDataArray; kw...) = pda[sortperm(pda; kw...)]

type FastPerm{O<:Base.Sort.Ordering,V<:AbstractVector} <: Base.Sort.Ordering
    ord::O
    vec::V
end
Base.sortperm{V}(x::AbstractVector, a::Base.Sort.Algorithm, o::FastPerm{Base.Sort.ForwardOrdering,V}) = x[sortperm(o.vec)]
Base.sortperm{V}(x::AbstractVector, a::Base.Sort.Algorithm, o::FastPerm{Base.Sort.ReverseOrdering,V}) = x[reverse(sortperm(o.vec))]
Perm{O<:Base.Sort.Ordering}(o::O, v::PooledDataVector) = FastPerm(o, v)

=#
#=
function PooledDataVecs{S,Q<:Integer,R<:Integer,N}(v1::PooledDataArray{S,Q,N},
                                                   v2::PooledDataArray{S,R,N})
    pool = sort(unique([v1.pool; v2.pool]))
    sz = length(pool)

    REFTYPE = sz <= typemax(UInt8)  ? UInt8 :
              sz <= typemax(UInt16) ? UInt16 :
              sz <= typemax(UInt32) ? UInt32 :
                                      UInt64

    tidx1 = convert(Vector{REFTYPE}, findat(pool, v1.pool))
    tidx2 = convert(Vector{REFTYPE}, findat(pool, v2.pool))
    refs1 = zeros(REFTYPE, length(v1))
    refs2 = zeros(REFTYPE, length(v2))
    for i in 1:length(refs1)
        if v1.refs[i] != 0
            refs1[i] = tidx1[v1.refs[i]]
        end
    end
    for i in 1:length(refs2)
        if v2.refs[i] != 0
            refs2[i] = tidx2[v2.refs[i]]
        end
    end
    return (PooledDataArray(RefArray(refs1), pool),
            PooledDataArray(RefArray(refs2), pool))
end

function PooledDataVecs{S,R<:Integer,N}(v1::PooledDataArray{S,R,N},
                                        v2::AbstractArray{S,N})
    return PooledDataVecs(v1,
                          PooledDataArray(v2))
end

####
function PooledDataVecs{S,R<:Integer,N}(v1::AbstractArray{S,N},
                                        v2::PooledDataArray{S,R,N})
    return PooledDataVecs(PooledDataArray(v1),
                          v2)
end

function PooledDataVecs(v1::AbstractArray,
                        v2::AbstractArray)

    ## Return two PooledDataVecs that share the same pool.

    ## TODO: allow specification of REFTYPE
    refs1 = Array(DEFAULT_POOLED_REF_TYPE, size(v1))
    refs2 = Array(DEFAULT_POOLED_REF_TYPE, size(v2))
    poolref = Dict{promote_type(eltype(v1), eltype(v2)), DEFAULT_POOLED_REF_TYPE}()
    maxref = 0

    # loop through once to fill the poolref dict
    for i = 1:length(v1)
        if !isna(v1[i])
            poolref[v1[i]] = 0
        end
    end
    for i = 1:length(v2)
        if !isna(v2[i])
            poolref[v2[i]] = 0
        end
    end

    # fill positions in poolref
    pool = sort(collect(keys(poolref)))
    i = 1
    for p in pool
        poolref[p] = i
        i += 1
    end

    # fill in newrefs
    zeroval = zero(DEFAULT_POOLED_REF_TYPE)
    for i = 1:length(v1)
        if isna(v1[i])
            refs1[i] = zeroval
        else
            refs1[i] = poolref[v1[i]]
        end
    end
    for i = 1:length(v2)
        if isna(v2[i])
            refs2[i] = zeroval
        else
            refs2[i] = poolref[v2[i]]
        end
    end

    return (PooledDataArray(RefArray(refs1), pool),
            PooledDataArray(RefArray(refs2), pool))
end
=#

##############################################################################
##
## conversions
##
##############################################################################

Base.convert{S,T,R1<:Integer,R2<:Integer,N}(::Type{PooledArray{S,R1,N}}, pa::PooledArray{T,R2,N}) =
    PooledArray(RefArray(convert(Array{R1,N}, pa.refs)), convert(Vector{S}, pa.pool))
Base.convert{S,T,R<:Integer,N}(::Type{PooledArray{S,R,N}}, pa::PooledArray{T,R,N}) =
    PooledArray(RefArray(copy(pa.refs)), convert(Vector{S}, pa.pool))
Base.convert{T,R<:Integer,N}(::Type{PooledArray{T,R,N}}, pa::PooledArray{T,R,N}) = pa
Base.convert{S,T,R1<:Integer,R2<:Integer,N}(::Type{PooledArray{S,R1}}, pa::PooledArray{T,R2,N}) =
    convert(PooledArray{S,R1,N}, pa)
Base.convert{S,T,R<:Integer,N}(::Type{PooledArray{S}}, pa::PooledArray{T,R,N}) =
    convert(PooledArray{S,R,N}, pa)
Base.convert{T,R<:Integer,N}(::Type{PooledArray}, pa::PooledArray{T,R,N}) = pa

Base.convert{S,T,R<:Integer,N}(::Type{PooledArray{S,R,N}}, a::AbstractArray{T,N}) =
    PooledArray(convert(Array{S,N}, a), R)
Base.convert{S,T,R<:Integer,N}(::Type{PooledArray{S,R}}, a::AbstractArray{T,N}) =
    PooledArray(convert(Array{S,N}, a), R)
Base.convert{S,T,N}(::Type{PooledArray{S}}, a::AbstractArray{T,N}) =
    PooledArray(convert(Array{S,N}, a))
Base.convert(::Type{PooledArray}, a::AbstractArray) =
    PooledArray(a)

function Base.convert{S, T, R, N}(::Type{Array{S, N}}, pa::PooledArray{T, R, N})
    res = Array(S, size(pa))
    for i in 1:length(pa)
        res[i] = pa.pool[pa.refs[i]]
    end
    return res
end

Base.convert{T, R}(::Type{Vector}, pv::PooledVector{T, R}) =
    convert(Array{T, 1}, pv)

Base.convert{T, R}(::Type{Matrix}, pm::PooledMatrix{T, R}) =
    convert(Array{T, 2}, pm)

Base.convert{T, R, N}(::Type{Array}, pa::PooledArray{T, R, N}) =
    convert(Array{T, N}, pa)

##############################################################################
##
## indexing
##
##############################################################################

# Scalar case
function Base.getindex(pa::PooledArray, I::Integer)
    return pa.pool[getindex(pa.refs, I)]
end
function Base.getindex(pa::PooledArray, I::Integer...)
    return pa.pool[getindex(pa.refs, I...)]
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

function Base.setindex!(x::PooledArray, val, ind::Integer)
    x.refs[ind] = getpoolidx(x, val)
    return x
end

##############################################################################
##
## growing and shrinking
##
##############################################################################

function Base.push!{S,R,T}(pv::PooledVector{S,R}, v::T)
    v = convert(S,v)
    push!(pv.refs, getpoolidx(pv, v))
    return v
end

Base.pop!(pv::PooledVector) = pv.pool[pop!(pv.refs)]

function Base.unshift!{S,R,T}(pv::PooledVector{S,R}, v::T)
    v = convert(S,v)
    unshift!(pv.refs, getpoolidx(pv, v))
    return v
end

Base.shift!(pv::PooledVector) = pv.pool[shift!(pv.refs)]

Base.empty!(pv::PooledVector) = (empty!(pv.refs); pv)

end
