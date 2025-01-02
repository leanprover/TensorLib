/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan, Paul Govereau, Sean McLaughlin
-/

import TensorLib.Common
import TensorLib.Tensor
import TensorLib.Slice
import TensorLib.Index

namespace TensorLib

/-
Shape of the resulting tensor is |slices| :: slices.map(fun x => x.shape)

Slices are treated differently in NumPy when used in mgrid vs indexing. In particular,
since we don't know what the stopping point should be from the context, it can use the `start`
value as a `stop` instead.  For example,

  # np.arange(10)[5:]
  [5, 6, 7, 8, 9]

  # np.mgrid[5:]
  [0, 1, 2, 3, 4]

However, things work normally when there is a stopping point

  # np.mgrid[5:10]
  [5, 6, 7, 8, 9]

Since this is imo surprising, we just fail if the start or stopping point are absent.
Moreover, we require more than one slice, since `mgrid[slice] == arange(slice)`, and a single
slice behaves differently from all other quantities.

All mgrid ints are stored as 64-bit, little-endian ints by convention. In Numpy they are stored with
native byte order, as the architecture word size.

For eample,

# np.mgrid[2:4:, 4:7:]
array([[[2, 2, 2],
        [3, 3, 3]],

       [[4, 5, 6],
        [4, 5, 6]]])

We have an iterator that will give us the indices [0, 0], [0, 1], [0, 2], [1, 0], ..., [1, 2]
and an interator that will give us the values [2, 4], [2, 5], [2, 6], ... , [3, 6]
These are the same length, so we can combine them to get the mgrid;
- [0, 0] of the first element gits 2,
- [0, 0] of the second element gets 4
- [0, 1] of the first gets 2
- [0, 1] of the second gets 5
- ...
-/
def mgrid (slices : List Slice) : Err Tensor := do
  let sliceCount := slices.length
  if sliceCount < 2 then .error "mgrid requires at least two slices"
  let arbitrary : Nat := 10  -- Slice.size does not use the second argument if both start and stop are specified
  let sliceSize (slice : Slice) : Nat := slice.size arbitrary
  let mut slicesDims := []
  for slice in slices.reverse do
    match slice.start, slice.stop with
    | .none, _ => .error "Slices need an upper bound in mgrid"
    | _, .none => .error "Slices need a lower bound in mgrid"
    | _, _ =>
      let sz := sliceSize slice
      slicesDims := sz :: slicesDims
  let shape := Shape.mk $ sliceCount :: slicesDims
  let slicesShape := Shape.mk slicesDims
  let dtype := Dtype.uint64
  let mut arr := Tensor.zeros dtype shape
  let basic := slices.map fun s => .slice (Slice.Iter.make s arbitrary)
  let mut sliceIter <- Index.BasicIter.make slicesShape basic
  let indexIter := DimsIter.make slicesShape
  if sliceIter.size != indexIter.size then .error "Invariant failure: iterator size mismatch at start"
  for index in indexIter do
    match sliceIter.next with
    | .none => .error "Invariant failure: iterator size mismatch during iteration"
    | .some (values, sliceIter') =>
      sliceIter := sliceIter'
      if values.length != sliceCount then .error "Invariant failure: value length mismatch"
      for (i, v) in (List.range sliceCount).zip values do
        let value := BV64.ofNat v
        arr <- Tensor.Element.setDimIndex arr (i :: index) value
  return arr

section Test

open TensorLib.Tensor.Format
open Tree

#guard (get! (mgrid [Slice.ofStartStop 2 4, Slice.ofStartStop 4 7])).toTree BV64 == .ok (
  .node [
    .node [
      .root [2, 2, 2], .root [3, 3, 3]
    ],
    .node [
      .root [4, 5, 6], .root [4, 5, 6]
    ]
  ]
)

#guard (get! (mgrid [Slice.ofStartStop 2 4, Slice.ofStartStop 4 7, Slice.ofStop 2])).toTree BV64 == .ok (
  .node [
    .node [
      .node [
        .root [2, 2], .root [2, 2], .root [2, 2]
      ],
      .node [
        .root [3, 3], .root [3, 3], .root [3, 3]
      ],
    ],
    .node [
      .node [
        .root [4, 4], .root [5, 5], .root [6, 6]
      ],
      .node [
        .root [4, 4], .root [5, 5], .root [6, 6]
      ]
    ],
    .node [
      .node [
        .root [0, 1], .root [0, 1], .root [0, 1]
      ],
      .node [
        .root [0, 1], .root [0, 1], .root [0, 1]
      ]
    ]
  ]
)

end Test
end TensorLib
