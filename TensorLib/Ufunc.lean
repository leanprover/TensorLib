import TensorLib.NumpyRepr
import TensorLib.Format

namespace TensorLib
namespace NumpyRepr

open TensorLib.NumpyRepr

namespace Ufunc

-- TODO: Add output array so we can clobber existing array. Will require reasoning about aliasing.
-- TODO: Add broadcasting logic to the method? Probably, and abstract similar to ufuncs in NumPy
private def binop (a : Type) [TensorElement a] (x y : NumpyRepr) (op : a -> a -> Err a) : Err NumpyRepr :=
  let dtype := x.dtype
  match broadcast { left := x.shape, right := y.shape } with
  | .none => .error s!"Can't broadcast shapes ${x.shape} with {y.shape}"
  | .some shape =>
    if dtype != y.dtype then .error s!"Casting between dtypes is not implemented {repr x.dtype} <> {repr y.dtype}" else
    let foldFn (acc : NumpyRepr) (idx : DimIndex) : Err NumpyRepr := do
      let v <- Index.getDimIndex x idx
      let w <- Index.getDimIndex y idx
      let k <- op v w
      Index.setDimIndex acc idx k
    shape.allDimIndices.foldlM foldFn (empty dtype shape)

def add (a : Type) [Add a] [TensorElement a] (x y : NumpyRepr) : Err NumpyRepr := binop a x y (fun x y => .ok (x + y))

#eval
  let x := (TensorElement.arange BV8 10)
  let arr := add BV8 x x
  match arr with
  | .error m => s!"Error: {m}"
  | .ok arr => Format.reprToString BV8 arr

#eval
  let x := (TensorElement.arange BV8 10)
  let y := TensorElement.arrayScalar (7 : BV8)
  let arr := add BV8 x y
  match arr with
  | .error m => s!"Error: {m}"
  | .ok arr => Format.reprToString BV8 arr

/-!
  """NKI kernel to compute element-wise addition of two input tensors

  This kernel assumes strict input/output tile-sizes, of up-to [128,512]

  Args:
      a_input: a first input tensor, of shape [128,512]
      b_input: a second input tensor, of shape [128,512]
      c_output: an output tensor, of shape [128,512]
  """
-/
private def nki_tensor_add_kernel_ (program_id0 program_id1 : Nat) (a_input b_input c_input : NumpyRepr) : Err Unit := do
  let tp := BV64

  -- Calculate tile offsets based on current 'program'
  let offset_i_x : tp := program_id0 * 128
  let offset_i_y : tp := program_id1 * 512

  -- Generate tensor indices to index tensors a and b
  let rx0 := TensorElement.arange tp 128
  let rx <- rx0.reshape [128, 1]
  let ox := TensorElement.arrayScalar offset_i_x
  let ix <- Ufunc.add tp ox rx
  let ry0 := TensorElement.arange tp 128
  let ry <- ry0.reshape [1, 512]
  let oy := TensorElement.arrayScalar offset_i_y
  let iy <- Ufunc.add tp oy ry

  let a_tile <- sorry -- load from a_input
  let b_tile <- sorry -- load from b_input

  let c_tile <- Ufunc.add tp a_tile b_tile

  let () <- sorry -- store to c_input

  .ok ()



end Ufunc
