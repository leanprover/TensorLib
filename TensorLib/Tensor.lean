import TensorLib.TensorData

namespace TensorLib

structure Tensor (a: Type) where
  data: TensorData a
  shape: Shape
  -- TODO: Currently `shape` is just a cache. It may be good to add this to the
  -- structure, but maintaining it is some work. Punt on this for now.
  -- hasShape: TensorData.HasShape a data shape

namespace Tensor

section Arith

variable [addI: Add a][subI: Sub a][mulI: Mul a][divI: Div a][negI: Neg a]
         [tensorElementI: TensorElement a]

def add (t1: Tensor a) (t2: Tensor a): Option (Tensor a) :=
  if t1.shape != t2.shape then none else do
    let d <- TensorData.add _ t1.data t2.data
    return Tensor.mk d t1.shape

end Arith

end Tensor
