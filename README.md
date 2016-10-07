# PooledArrays.jl
A pooled representation of arrays for purposes of compression when there are few unique elements.

This implementation is designed for elements with a total order. The pool of unique values is
maintained in sorted order, allowing efficient comparison and sorting based on integer IDs.
