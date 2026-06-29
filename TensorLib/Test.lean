/-
Copyright TensorLib Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-/

import Init.System.IO
import TensorLib.Tensor

namespace TensorLib

namespace Test

-- Caller must remove the temp file
private def saveNumpyArray (expr : String) : IO System.FilePath := do
  let (_root_, file) <- IO.FS.createTempFile
  let expr := s!"import numpy as np; x = {expr}; np.save('{file}', x)"
  let _output <- IO.Process.output { cmd := "/usr/bin/env", args := ["python3", "-c", expr].toArray }
  -- `np.save` appends `.npy` to the file
  return file.addExtension "npy"

private def testTensorElementBV (dtype : Dtype) : IO Bool := do
  let file <- saveNumpyArray s!"np.arange(20, dtype='{dtype}').reshape(5, 4)"
  let npy <- Npy.parseFile file
  let arr <- IO.ofExcept (Tensor.ofNpy npy)
  let _ <- IO.FS.removeFile file
  let expected := (Tensor.arange! dtype 20).reshape! (Shape.mk [5, 4])
  return Tensor.arrayEqual expected arr

-- fp16 edge cases decode corrctly after loading from npy
-- Each value is input as fp16 bytes, decoded to fp32, and compared against expected
private def testFloat16EdgeCases : IO Bool := do
    let file <- saveNumpyArray "np.array([65504.0, 0.1, -0.0, np.inf, -np.inf, np.nan, 2048.5, 100.5, 0.00005, 1.16e-10, 1e-20, -1e-20, 1e-40], dtype='float16')"
    let npy <- Npy.parseFile file
    let arr <- IO.ofExcept (Tensor.ofNpy npy)
    let _ <- IO.FS.removeFile file
    -- decode a 2 byte fp16 value at index x 2 since each fp16 is 2 bytes
    let decode (offset : Nat) : Err Float32 :=
      Dtype.byteArrayToFloat16 .float16 (arr.data.extract offset (offset + 2))
    let posInf := Float32.ofBits 0x7F800000
    let negInf := Float32.ofBits 0xFF800000
    -- max in fp16
    let v0 <- IO.ofExcept (decode 0)
    let c0 := v0 == 65504.0
    IO.println s!"v0 (65504): {v0 == 65504.0}"
    -- 0.1 cant be exact so check approx
    let v1 <- IO.ofExcept (decode 2)
    let diff := v1 - 0.1
    let c1 := diff < 0.002 && diff > -0.002
    IO.println s!"v1 (0.1): {c1}"
    -- IEE754 -0.0 == 0.0
    let v2 <- IO.ofExcept (decode 4)
    let c2 := v2 == 0.0
    IO.println s!"v2 (-0): {c2}"
    -- +inf, -inf
    let v3 <- IO.ofExcept (decode 6)
    let c3 := v3 == posInf
    IO.println s!"v3 (inf): {c3}"
    let v4 <- IO.ofExcept (decode 8)
    let c4 := v4 == negInf
    IO.println s!"v4 (-inf): {c4}"
    -- nan != nan
    let v5 <- IO.ofExcept (decode 10)
    let c5 := v5 != v5
    IO.println s!"v5 (nan): {c5}"
    -- 2048.5 rounds to 2048 in fp16 npy (step size is 2)
    let v6 <- IO.ofExcept (decode 12)
    IO.println s!"v6 actual: {v6}"
    let c6 := v6 == 2048
    IO.println s!"v6 (2048.5 rounds to 2048): {c6}"
    -- 100.5 fits in fp16
    let v7 <- IO.ofExcept (decode 14)
    let c7 := v7 == 100.5
    IO.println s!"v7 (100.5) : {c7}"
    -- subnormals rounding
    let v8 <- IO.ofExcept (decode 16)
    let diff8 := v8 - 0.00005
    let c8 :=  diff8 < 0.000001 && diff8 > -0.000001
    IO.println s!"v8 (subnormal 0.00005): {c8}, actual: {v8}"
    -- below fp16 minimum subnormal so should be 0
    let v9 <- IO.ofExcept (decode 18)
    let c9 := v9 == 0.0
    IO.println s!"v9 (1.16e-10 -> 0): {c9}"
    -- more values below fp16 min subnormal → zero
    let v10 <- IO.ofExcept (decode 20)
    let c10 := v10 == 0.0
    IO.println s!"v10 (1e-20 -> 0): {c10}"
    let v11 <- IO.ofExcept (decode 22)
    let c11 := v11 == 0.0  -- -0.0 == 0.0 per IEEE 754
    IO.println s!"v11 (-1e-20 -> -0): {c11}"
    let v12 <- IO.ofExcept (decode 24)
    let c12 := v12 == 0.0
    IO.println s!"v12 (1e-40 -> 0): {c12}"
    return c0 && c1 && c2 && c3 && c4 && c5 && c6 && c7 && c8 && c9 && c10 && c11 && c12



def runAllTests : IO Bool := do
 return (<- testTensorElementBV Dtype.uint16) &&
        (<- testTensorElementBV Dtype.uint32) &&
        (<- testFloat16EdgeCases)

end Test
end TensorLib
