import TensorLib.Common
import TensorLib.Tensor
import TensorLib.Broadcast

namespace TensorLib
namespace Tensor
namespace Ufunc

private def binop (a : Type) [Element a] (x y : Tensor) (op : a -> a -> Err a) : Err Tensor :=
  match Broadcast.broadcast { left := x.shape, right := y.shape } with
  | .none => .error s!"Can't broadcast shapes ${x.shape} with {y.shape}"
  | .some shape =>
    if x.dtype != y.dtype then .error s!"Casting between dtypes is not implemented yet: {repr x.dtype} <> {repr y.dtype}" else
    do
      let mut arr := Tensor.empty x.dtype shape
      let iter := DimsIter.make shape
      for idx in iter do
        let v <- Element.getDimIndex x idx
        let w <- Element.getDimIndex y idx
        let k <- op v w
        let arr' <- Element.setDimIndex arr idx k
        arr := arr'
      .ok arr

def add (a : Type) [Add a] [Element a] (x y : Tensor) : Err Tensor :=
  binop a x y (fun x y => .ok (x + y))

def sub (a : Type) [Sub a] [Element a] (x y : Tensor) : Err Tensor :=
  binop a x y (fun x y => .ok (x - y))

def mul (a : Type) [Mul a] [Element a] (x y : Tensor) : Err Tensor :=
  binop a x y (fun x y => .ok (x * y))

def div (a : Type) [Div a] [Element a] (x y : Tensor) : Err Tensor :=
  binop a x y (fun x y => .ok (x / y))

/-
TODO:
- np.sum. Prove that np.sum(x, axis=(2, 4, 6)) == np.sum(np.sum(np.sum(x, axis=6), axis=4), axis=2) # and other variations
-/

#eval
  let x := (Element.arange BV8 10)
  let arr := add BV8 x x
  match arr with
  | .error m => s!"Error: {m}"
  | .ok arr => arr.str BV8

#eval
  let x := (Element.arange BV8 10)
  let y := Element.arrayScalar (7 : BV8)
  let arr := add BV8 x y
  match arr with
  | .error m => s!"Error: {m}"
  | .ok arr => arr.str BV8

/-!
  """NKI kernel to compute element-wise addition of two input tensors

  This kernel assumes strict input/output tile-sizes, of up-to [128,512]

  Args:
      a_input: a first input tensor, of shape [128,512]
      b_input: a second input tensor, of shape [128,512]
      c_output: an output tensor, of shape [128,512]
  """
-/
-- private def nki_tensor_add_kernel_ (program_id0 program_id1 : Nat) (a_input b_input c_input : NumpyRepr) : Err Unit := do
--   let tp := BV64

--   -- Calculate tile offsets based on current 'program'
--   let offset_i_x : tp := program_id0 * 128
--   let offset_i_y : tp := program_id1 * 512

--   -- Generate tensor indices to index tensors a and b
--   let rx0 := Element.arange tp 128
--   let rx <- rx0.reshape [128, 1]
--   let ox := Element.arrayScalar offset_i_x
--   let ix <- Ufunc.add tp ox rx
--   let ry0 := Element.arange tp 128
--   let ry <- ry0.reshape [1, 512]
--   let oy := Element.arrayScalar offset_i_y
--   let iy <- Ufunc.add tp oy ry

--   let a_tile <- sorry -- load from a_input
--   let b_tile <- sorry -- load from b_input

--   let c_tile <- Ufunc.add tp a_tile b_tile

--   let () <- sorry -- store to c_input
--   .ok ()



end Ufunc
