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
    IO.println s!"fp16 v0 (65504): {v0 == 65504.0}"
    -- 0.1 cant be exact so check approx
    let v1 <- IO.ofExcept (decode 2)
    let diff := v1 - 0.1
    let c1 := diff < 0.002 && diff > -0.002
    IO.println s!"fp16 v1 (0.1): {c1}"
    -- IEE754 -0.0 == 0.0
    let v2 <- IO.ofExcept (decode 4)
    let c2 := v2 == 0.0
    IO.println s!"fp16 v2 (-0): {c2}"
    -- +inf, -inf
    let v3 <- IO.ofExcept (decode 6)
    let c3 := v3 == posInf
    IO.println s!"fp16 v3 (inf): {c3}"
    let v4 <- IO.ofExcept (decode 8)
    let c4 := v4 == negInf
    IO.println s!"fp16 v4 (-inf): {c4}"
    -- nan != nan
    let v5 <- IO.ofExcept (decode 10)
    let c5 := v5 != v5
    IO.println s!"fp16 v5 (nan): {c5}"
    -- 2048.5 rounds to 2048 in fp16 npy (step size is 2)
    let v6 <- IO.ofExcept (decode 12)
    IO.println s!"fp16 v6 actual: {v6}"
    let c6 := v6 == 2048
    IO.println s!"fp16 v6 (2048.5 rounds to 2048): {c6}"
    -- 100.5 fits in fp16
    let v7 <- IO.ofExcept (decode 14)
    let c7 := v7 == 100.5
    IO.println s!"fp16 v7 (100.5) : {c7}"
    -- subnormals rounding
    let v8 <- IO.ofExcept (decode 16)
    let diff8 := v8 - 0.00005
    let c8 :=  diff8 < 0.000001 && diff8 > -0.000001
    IO.println s!"fp16 v8 (subnormal 0.00005): {c8}, actual: {v8}"
    -- below fp16 minimum subnormal so should be 0
    let v9 <- IO.ofExcept (decode 18)
    let c9 := v9 == 0.0
    IO.println s!"fp16 v9 (1.16e-10 -> 0): {c9}"
    -- more values below fp16 min subnormal → zero
    let v10 <- IO.ofExcept (decode 20)
    let c10 := v10 == 0.0
    IO.println s!"fp16 v10 (1e-20 -> 0): {c10}"
    let v11 <- IO.ofExcept (decode 22)
    let c11 := v11 == 0.0  -- -0.0 == 0.0 per IEEE 754
    IO.println s!"fp16 v11 (-1e-20 -> -0): {c11}"
    let v12 <- IO.ofExcept (decode 24)
    let c12 := v12 == 0.0
    IO.println s!"fp16 v12 (1e-40 -> 0): {c12}"
    return c0 && c1 && c2 && c3 && c4 && c5 && c6 && c7 && c8 && c9 && c10 && c11 && c12


-- bf16 edge cases
private def testBFloat16EdgeCases : IO Bool := do
  let file <- saveNumpyArray  "np.array([256.0, 0.1, -0.0, np.inf, -np.inf, np.nan, 3.3895e38]).astype(__import__('ml_dtypes').bfloat16)"
  let npy <- Npy.parseFile file
  let arr <- IO.ofExcept (Tensor.ofNpy npy)
  let _ <- IO.FS.removeFile file
  let decode (offset : Nat) : Err Float32 :=
    Dtype.byteArrayToBFloat16 .bfloat16 (arr.data.extract offset (offset + 2))
  let v0 <- IO.ofExcept (decode 0)
  let c0 := v0 == 256.0
  IO.println s!"bf16 v0 (256): {c0}"
  -- 0.1 rounded in bf16
  let v1 <- IO.ofExcept (decode 2)
  let diff := v1 - 0.1
  let c1 := diff < 0.01 && diff > -0.01
  IO.println s!"bf16 v1 (0.1): {c1}"
  -- negative zero
  let v2 <- IO.ofExcept (decode 4)
  let c2 := v2 == 0.0
  IO.println s!"bf16 v2 (-0): {c2}"
  -- +inf
  let v3 <- IO.ofExcept (decode 6)
  let c3 := v3 == Float32.ofBits 0x7F800000
  IO.println s!"bf16 v3 (inf): {c3}"
  -- -inf
  let v4 <- IO.ofExcept (decode 8)
  let c4 := v4 == Float32.ofBits 0xFF800000
  IO.println s!"bf16 v4 (-inf): {c4}"
  -- NaN
  let v5 <- IO.ofExcept (decode 10)
  let c5 := v5 != v5
  IO.println s!"bf16 v5 (NaN): {c5}"
  -- max bf16 value
  let v6 <- IO.ofExcept (decode 12)
  let c6 := v6 == Float32.ofBits 0x7F7F0000
  IO.println s!"bf16 v6 (max): {c6}"
  -- Arithmetic correctness: compare against ml_dtypes results
  let a := toLEByteArray (Float32.ofNat 3 / Float32.ofNat 2).toBFloat16Bits  -- bf16 encoding of 1.5
  let b := toLEByteArray (Float32.ofNat 5 / Float32.ofNat 2).toBFloat16Bits  -- bf16 encoding of 2.5
  -- 1.5 + 2.5 = 4.0, ml_dtypes gives bits 16512
  let addResult <- IO.ofExcept (Dtype.add .bfloat16 a b)
  let c7 := addResult == toLEByteArray (16512 : UInt16)
  IO.println s!"bf16 add (1.5 + 2.5 = 4.0): {c7}"
  -- 1.5 - 2.5 = -1.0, ml_dtypes gives bits 49024
  let subResult <- IO.ofExcept (Dtype.sub .bfloat16 a b)
  let c8 := subResult == toLEByteArray (49024 : UInt16)
  IO.println s!"bf16 sub (1.5 - 2.5 = -1.0): {c8}"
  -- 1.5 * 2.5 = 3.75, ml_dtypes gives bits 16496
  let mulResult <- IO.ofExcept (Dtype.mul .bfloat16 a b)
  let c9 := mulResult == toLEByteArray (16496 : UInt16)
  IO.println s!"bf16 mul (1.5 * 2.5 = 3.75): {c9}"
  -- 1.5 / 2.5 = 0.6, ml_dtypes gives bits 16154
  let divResult <- IO.ofExcept (Dtype.div .bfloat16 a b)
  let c10 := divResult == toLEByteArray (16154 : UInt16)
  IO.println s!"bf16 div (1.5 / 2.5 = 0.6): {c10}"
  -- abs(-1.5) = 1.5, ml_dtypes gives bits 16320
  let negA := toLEByteArray (Float32.ofNat 3 / Float32.ofNat 2).neg.toBFloat16Bits  -- bf16 encoding of -1.5
  let absResult <- IO.ofExcept (Dtype.abs .bfloat16 negA)
  let c11 := absResult == toLEByteArray (16320 : UInt16)
  IO.println s!"bf16 abs (-1.5) =  1.5: {c11}"
  -- Casting test cases
  -- bf16(42) -> float32 should give 42.0
  let bf42 := toLEByteArray (Float32.ofNat 42).toBFloat16Bits  -- bf16 encoding of 42
  let castToF32 <- IO.ofExcept (Dtype.castOverflow .bfloat16 bf42 .float32)
  let f32Val <- IO.ofExcept (Float32.ofLEByteArray castToF32)
  let c12 := f32Val == 42.0
  IO.println s!"bf16 cast bf16 to f32 (42): {c12}"
  -- bf16(42) -> int8 should give 42
  let castToI8 <- IO.ofExcept (Dtype.castOverflow .bfloat16 bf42 .int8)
  let c13 := castToI8.toNat == 42
  IO.println s!"bf16 cast bf16 to int8 (42): {c13}"
  -- f32(3.14) -> bf16 should give bits 16457
  let f32_314 := toLEByteArray (Float32.ofNat 314 / Float32.ofNat 100)
  let castToBf16 <- IO.ofExcept (Dtype.castOverflow .float32 f32_314 .bfloat16)
  let c14 := castToBf16 == toLEByteArray (16457 : UInt16)
  IO.println s!"bf16 fp32 to bf16 (3.14): {c14}"
  -- float16(1.5) -< bfloat16 should give bits 16320
  let f16_15 := toLEByteArray (Float32.ofNat 3 / Float32.ofNat 2).toFloat16Bits  -- fp16 encoding of 1.5
  let castF16ToBf16 <- IO.ofExcept (Dtype.castOverflow .float16 f16_15 .bfloat16)
  let c15 := castF16ToBf16 == toLEByteArray (16320 : UInt16)
  IO.println s!"bf16 fp16 to bf16 (1.5): {c15}"
  -- bf16(-0.0) -> bool should give 0 (false)
  -- Tests isFloat/isZero: -0.0 has nonzero sign byte (0x80) but is still 0
  let bf16NegZero := toLEByteArray (0x8000 : UInt16)  -- bf16 -0.0
  let castToBool <- IO.ofExcept (Dtype.castOverflow .bfloat16 bf16NegZero .bool)
  let c16 := castToBool == ByteArray.mk #[0]
  IO.println s!"bf16 -0 to bool (false): {c16}"
   -- bf16(inf) to fp32 should give inf
  let bf16Inf := toLEByteArray (0x7F80 : UInt16)  -- bf16 +inf
  let castInfToF32 <- IO.ofExcept (Dtype.castOverflow .bfloat16 bf16Inf .float32)
  let infVal <- IO.ofExcept (Float32.ofLEByteArray castInfToF32)
  let c17 := infVal == Float32.ofBits 0x7F800000
  IO.println s!"bf16 inf to fp32: {c17}"


  return c0 && c1 && c2 && c3 && c4 && c5 && c6 && c7 && c8 && c9 && c10 && c11 && c12 && c13 && c14 && c15 && c16 && c17




def runAllTests : IO Bool := do
 return (<- testTensorElementBV Dtype.uint16) &&
        (<- testTensorElementBV Dtype.uint32) &&
        (<- testFloat16EdgeCases) &&
        (<- testBFloat16EdgeCases)

end Test
end TensorLib
