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
  let output <- IO.Process.output { cmd := "/usr/bin/env", args := ["python3", "-c", expr].toArray }
  -- Fail fast with a clear message if Python or a dependency (e.g. ml_dtypes) is missing
  if output.exitCode != 0 then throw $ IO.userError s!"Python failed (exit {output.exitCode}): {output.stderr}"
  -- `np.save` appends `.npy` to the file
  return file.addExtension "npy"

-- Helper to check that a dtype operation produced the expected UInt16 bit pattern
  private def checkBits (label : String) (expected : UInt16) (act : Err ByteArray) : IO Bool := do
    let result <- IO.ofExcept act
    let pass := result == toLEByteArray expected
    IO.println s!"{label}: {pass}"
    return pass

private def testTensorElementBV (dtype : Dtype) : IO Bool := do
  let file <- saveNumpyArray s!"np.arange(20, dtype='{dtype}').reshape(5, 4)"
  let npy <- Npy.parseFile file
  let arr <- IO.ofExcept (Tensor.ofNpy npy)
  let _ <- IO.FS.removeFile file
  let expected := (Tensor.arange! dtype 20).reshape! (Shape.mk [5, 4])
  return Tensor.arrayEqual expected arr

private def testFloat16EdgeCases : IO Bool := do
  let file <- saveNumpyArray "np.array([65504.0, 0.1, -0.0, np.inf, -np.inf, np.nan, 2048.5, 100.5, 0.00005, 1.16e-10, 1e-20, -1e-20, 1e-40], dtype='float16')"
  let npy <- Npy.parseFile file
  let arr <- IO.ofExcept (Tensor.ofNpy npy)
  let _ <- IO.FS.removeFile file
  let decode (offset : Nat) : Err Float32 :=
    Dtype.byteArrayToFloat16 .float16 (arr.data.extract offset (offset + 2))
  let posInf := Float32.ofBits 0x7F800000
  let negInf := Float32.ofBits 0xFF800000
  let mut checks : List Bool := []
  let mut pass := true

  -- max in fp16
  let v0 <- IO.ofExcept (decode 0)
  pass := v0 == Float32.ofBits 0x477FE000
  IO.println s!"fp16 v0 (65504): {pass}"
  checks := checks ++ [pass]

  -- 0.1 cant be exact so check approx
  let v1 <- IO.ofExcept (decode 2)
  let diff := v1 - 0.1
  pass := diff < 0.002 && diff > -0.002
  IO.println s!"fp16 v1 (0.1): {pass}"
  checks := checks ++ [pass]

  -- IEEE754 -0.0 == 0.0
  let v2 <- IO.ofExcept (decode 4)
  pass := v2 == 0.0
  IO.println s!"fp16 v2 (-0): {pass}"
  checks := checks ++ [pass]

  -- +inf
  let v3 <- IO.ofExcept (decode 6)
  pass := v3 == posInf
  IO.println s!"fp16 v3 (inf): {pass}"
  checks := checks ++ [pass]

  -- -inf
  let v4 <- IO.ofExcept (decode 8)
  pass := v4 == negInf
  IO.println s!"fp16 v4 (-inf): {pass}"
  checks := checks ++ [pass]

  -- nan != nan
  let v5 <- IO.ofExcept (decode 10)
  pass := v5 != v5
  IO.println s!"fp16 v5 (nan): {pass}"
  checks := checks ++ [pass]

  -- 2048.5 rounds to 2048 in fp16 npy (step size is 2)
  let v6 <- IO.ofExcept (decode 12)
  IO.println s!"fp16 v6 actual: {v6}"
  pass := v6 == Float32.ofNat 2048
  IO.println s!"fp16 v6 (2048.5 rounds to 2048): {pass}"
  checks := checks ++ [pass]

  -- 100.5 fits in fp16
  let v7 <- IO.ofExcept (decode 14)
  pass := v7 == 100.5
  IO.println s!"fp16 v7 (100.5) : {pass}"
  checks := checks ++ [pass]

  -- subnormals rounding
  let v8 <- IO.ofExcept (decode 16)
  let diff8 := v8 - 0.00005
  pass := diff8 < 0.000001 && diff8 > -0.000001
  IO.println s!"fp16 v8 (subnormal 0.00005): {pass}, actual: {v8}"
  checks := checks ++ [pass]

  -- below fp16 minimum subnormal so should be 0
  let v9 <- IO.ofExcept (decode 18)
  pass := v9 == 0.0
  IO.println s!"fp16 v9 (1.16e-10 -> 0): {pass}"
  checks := checks ++ [pass]

  -- more values below fp16 min subnormal → zero
  let v10 <- IO.ofExcept (decode 20)
  pass := v10 == 0.0
  IO.println s!"fp16 v10 (1e-20 -> 0): {pass}"
  checks := checks ++ [pass]

  let v11 <- IO.ofExcept (decode 22)
  pass := v11 == 0.0
  IO.println s!"fp16 v11 (-1e-20 -> -0): {pass}"
  checks := checks ++ [pass]

  let v12 <- IO.ofExcept (decode 24)
  pass := v12 == 0.0
  IO.println s!"fp16 v12 (1e-40 -> 0): {pass}"
  checks := checks ++ [pass]

  return checks.all id

-- bf16 edge cases
private def testBFloat16EdgeCases : IO Bool := do
  let file <- saveNumpyArray  "np.array([256.0, 0.1, -0.0, np.inf, -np.inf, np.nan, 3.3895e38]).astype(__import__('ml_dtypes').bfloat16)"
  let npy <- Npy.parseFile file
  let arr <- IO.ofExcept (Tensor.ofNpy npy)
  let _ <- IO.FS.removeFile file
  let decode (offset : Nat) : Err Float32 :=
    Dtype.byteArrayToBFloat16 .bfloat16 (arr.data.extract offset (offset + 2))
  let mut checks : List Bool := []

  let v0 <- IO.ofExcept (decode 0)
  let pass := v0 == 256.0
  IO.println s!"bf16 v0 (256): {pass}"
  checks := checks ++ [pass]

  -- 0.1 rounded in bf16
  let v1 <- IO.ofExcept (decode 2)
  let diff := v1 - 0.1
  let pass := diff < 0.01 && diff > -0.01
  IO.println s!"bf16 v1 (0.1): {pass}"
  checks := checks ++ [pass]

  -- -0
  let v2 <- IO.ofExcept (decode 4)
  let pass := v2 == 0.0
  IO.println s!"bf16 v2 (-0): {pass}"
  checks := checks ++ [pass]

  -- +inf
  let v3 <- IO.ofExcept (decode 6)
  let pass := v3 == Float32.ofBits 0x7F800000
  IO.println s!"bf16 v3 (inf): {pass}"
  checks := checks ++ [pass]

  -- -inf
  let v4 <- IO.ofExcept (decode 8)
  let pass := v4 == Float32.ofBits 0xFF800000
  IO.println s!"bf16 v4 (-inf): {pass}"
  checks := checks ++ [pass]

  -- NaN
  let v5 <- IO.ofExcept (decode 10)
  let pass := v5 != v5
  IO.println s!"bf16 v5 (NaN): {pass}"
  checks := checks ++ [pass]

  -- max bf16 value
  let v6 <- IO.ofExcept (decode 12)
  let pass := v6 == Float32.ofBits 0x7F7F0000
  IO.println s!"bf16 v6 (max): {pass}"
  checks := checks ++ [pass]

  -- Arithmetic correctness: compare against ml_dtypes results
  let a := toLEByteArray (Float32.ofNat 3 / Float32.ofNat 2).toBFloat16Bits  -- bf16 encoding of 1.5
  let b := toLEByteArray (Float32.ofNat 5 / Float32.ofNat 2).toBFloat16Bits  -- bf16 encoding of 2.5

  let pass <- checkBits "bf16 add (1.5 + 2.5 = 4.0)" 16512 (Dtype.add .bfloat16 a b)
  checks := checks ++ [pass]

  let pass <- checkBits "bf16 sub (1.5 - 2.5 = -1.0)" 49024 (Dtype.sub .bfloat16 a b)
  checks := checks ++ [pass]

  let pass <- checkBits "bf16 mul (1.5 * 2.5 = 3.75)" 16496 (Dtype.mul .bfloat16 a b)
  checks := checks ++ [pass]

  let pass <- checkBits "bf16 div (1.5 / 2.5 = 0.6)" 16154 (Dtype.div .bfloat16 a b)
  checks := checks ++ [pass]

  -- abs(-1.5) = 1.5, ml_dtypes gives bits 16320
  let negA := toLEByteArray (Float32.ofNat 3 / Float32.ofNat 2).neg.toBFloat16Bits
  let pass <- checkBits "bf16 abs (-1.5) = 1.5" 16320 (Dtype.abs .bfloat16 negA)
  checks := checks ++ [pass]

  -- Casting test cases
  -- bf16(42) -> float32 should give 42.0
  let bf42 := toLEByteArray (Float32.ofNat 42).toBFloat16Bits
  let castToF32 <- IO.ofExcept (Dtype.castOverflow .bfloat16 bf42 .float32)
  let f32Val <- IO.ofExcept (Float32.ofLEByteArray castToF32)
  let pass := f32Val == 42.0
  IO.println s!"bf16 cast bf16 to f32 (42): {pass}"
  checks := checks ++ [pass]

  -- bf16(42) -> int8 should give 42
  let castToI8 <- IO.ofExcept (Dtype.castOverflow .bfloat16 bf42 .int8)
  let pass := castToI8.toNat == 42
  IO.println s!"bf16 cast bf16 to int8 (42): {pass}"
  checks := checks ++ [pass]

  -- f32(3.14) -> bf16 should give bits 16457
  let f32_314 := toLEByteArray (Float32.ofNat 314 / Float32.ofNat 100)
  let pass <- checkBits "bf16 fp32 to bf16 (3.14)" 16457 (Dtype.castOverflow .float32 f32_314 .bfloat16)
  checks := checks ++ [pass]

  -- float16(1.5) -> bfloat16 should give bits 16320
  let f16_15 := toLEByteArray (Float32.ofNat 3 / Float32.ofNat 2).toFloat16Bits
  let pass <- checkBits "bf16 fp16 to bf16 (1.5)" 16320 (Dtype.castOverflow .float16 f16_15 .bfloat16)
  checks := checks ++ [pass]

  -- bf16(-0.0) -> bool should give 0 (false)
  let bf16NegZero := toLEByteArray (0x8000 : UInt16)
  let castToBool <- IO.ofExcept (Dtype.castOverflow .bfloat16 bf16NegZero .bool)
  let pass := castToBool == ByteArray.mk #[0]
  IO.println s!"bf16 -0 to bool (false): {pass}"
  checks := checks ++ [pass]

  -- bf16(inf) to fp32 should give inf
  let bf16Inf := toLEByteArray (0x7F80 : UInt16)
  let castInfToF32 <- IO.ofExcept (Dtype.castOverflow .bfloat16 bf16Inf .float32)
  let infVal <- IO.ofExcept (Float32.ofLEByteArray castInfToF32)
  let pass := infVal == Float32.ofBits 0x7F800000
  IO.println s!"bf16 inf to fp32: {pass}"
  checks := checks ++ [pass]

  -- Cast: bf16(-1.5) -> uint8 (NOTE: diverges from numpy which gives 255 via wrapping)
  -- Lean's Float32.toNat forces negatives to 0
  let negBf16 := toLEByteArray (Float32.ofNat 3 / Float32.ofNat 2).neg.toBFloat16Bits
  let castNegToU8 <- IO.ofExcept (Dtype.castOverflow .bfloat16 negBf16 .uint8)
  let pass := castNegToU8.toNat == 0
  IO.println s!"bf16 cast -1.5 to uint8 (clamped to 0): {pass}"
  checks := checks ++ [pass]

  -- Cast: bf16(inf) -> uint8 = 255 in both lean and numpy
  let bf16Inf := toLEByteArray (0x7F80 : UInt16)
  let castInfToU8 <- IO.ofExcept (Dtype.castOverflow .bfloat16 bf16Inf .uint8)
  let pass := castInfToU8.toNat == 255
  IO.println s!"bf16 cast inf to uint8 (255): {pass}"
  checks := checks ++ [pass]

  return checks.all id


def runAllTests : IO Bool := do
 return (<- testTensorElementBV Dtype.uint16) &&
        (<- testTensorElementBV Dtype.uint32) &&
        (<- testFloat16EdgeCases) &&
        (<- testBFloat16EdgeCases)

end Test
end TensorLib
