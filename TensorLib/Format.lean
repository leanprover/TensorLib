import TensorLib.Tensor
import TensorLib.Indexing

namespace TensorLib
namespace Format

-- open TensorLib.NumpyRepr
open Std.Format

-- TODO: remove `Err` by proving all indices are within range
private def toList (a : Type) [Tensor.Element a] (x : Tensor) : Err (List a) :=
  let traverseFn ind : Err a := Index.getDimIndex x ind
  traverse traverseFn x.shape.allDimIndices

private def toList! (a : Type) [Tensor.Element a] (x : Tensor) : List a := match toList a x with
| .error _ => []
| .ok x => x

#guard toList! BV16 (Tensor.Element.arange BV16 10) == [0x0000#16, 0x0001#16, 0x0002#16, 0x0003#16, 0x0004#16, 0x0005#16, 0x0006#16, 0x0007#16, 0x0008#16, 0x0009#16]

-- Useful for small arrays, e.g. to help with printing and such
-- There are some natural invariants we could check, such as that the
-- trees in a node all have the same height, but since this is just a
-- utility structure we'll keep it simple
private inductive Tree a where
| root (xs: List a)
| node (xs: List (Tree a))
deriving Repr

private def toTree {a : Type} (x : List a) (strides : Strides) : Err (Tree a) :=
  if strides.any fun x => x <= 0 then .error "strides need to be positive" else
  match strides with
  | [] => if x.length == 1 then .ok (.root x) else .error "empty shape that's not an array scalar"
  | [1] => .ok (.root x)
  | [_] => .error "not a unit stride"
  | stride :: strides => do
    let chunks := x.toChunks stride.toNat
    let res <- chunks.traverse (fun x => toTree x strides)
    return .node res

private def toTree! {a : Type} (x : List a) (strides : Strides) : Tree a := match toTree x strides with
| .error _ => .root []
| .ok t => t

#eval toTree! (toList! BV16 (Tensor.Element.arange BV16 10)) [5, 1]

private def formatRoot [Repr a] (xs : List a) : Std.Format := sbracket (joinSep (List.map repr xs) (text ", "))

private def formatTree1 [Repr a] (shape : Shape) (t : Tree a) : Err Std.Format := match shape, t with
| [], .root [x] => .ok $ repr x
| [n], .root r => if r.length != n then .error "shape mismatch" else .ok (formatRoot r)
| n :: shape, .node ts => do
  let fmts <- ts.traverse (formatTree1 shape)
  if fmts.length != n then .error "head mismatch" else
  let indented := join (fmts.intersperse (", " ++ line))
  .ok (group (nest 2 ("[" ++ indented ++ "]")))
| _, _ => .error "format mismatch"

/- This needs some improvement. For example, I'm not able to get the indent to stick
at the end of the "array("

$ bin/tensorlib format 20 2
Got shape [20, 2]
array([[0x0000#16, 0x0001#16],
  [0x0002#16, 0x0003#16],
  [0x0004#16, 0x0005#16],
  ...
-/
private def formatTree [Repr a] (t : Tree a) (shape : Shape) : Err Std.Format := do
  let r <- formatTree1 shape t
  return join ["array(", r, ")"]

def format (a : Type) [Repr a] [Tensor.Element a] (x : Tensor) : Err Std.Format := do
  let xs <- toList a x
  let t <- toTree xs x.unitStrides
  let f <- formatTree t x.shape
  return f

def reprToString (a : Type) [Repr a] [Tensor.Element a] (x : Tensor) : String := match format a x with
| .error err => s!"Error: {err}"
| .ok s => pretty s 120

#guard reprToString BV8 (Tensor.Element.arrayScalar (5 : BV8)) == "array(0x05#8)"
#guard reprToString BV8 (Tensor.Element.arange BV8 10) == "array([0x00#8, 0x01#8, 0x02#8, 0x03#8, 0x04#8, 0x05#8, 0x06#8, 0x07#8, 0x08#8, 0x09#8])"

end Format
