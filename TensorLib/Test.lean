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
  if output.exitCode != 0 then do
    IO.FS.removeFile file
    throw $ IO.userError s!"Python failed (exit {output.exitCode}): {output.stderr}"
  -- `np.save` appends `.npy` to the file
  return file.addExtension "npy"

-- Helper to check that a dtype operation produced the expected UInt16 bit pattern
private def checkBits (label : String) (expected : UInt16) (act : Err ByteArray) : IO Bool := do
  let result <- IO.ofExcept act
  let pass := result == toLEByteArray expected
  IO.println s!"{label}: {pass}"
  return pass

-- Helper: check that a dtype operation produced the expected UInt8 bit pattern
private def checkBitsU8 (label : String) (expected : UInt8) (act : Err ByteArray) : IO Bool := do
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
  -- let mut pass := true

  -- max in fp16
  let v0 <- IO.ofExcept (decode 0)
  let pass := v0 == Float32.ofBits 0x477FE000
  IO.println s!"fp16 v0 (65504): {pass}"
  checks := pass :: checks

  -- 0.1 cant be exact so check approx
  let v1 <- IO.ofExcept (decode 2)
  let diff := v1 - 0.1
  let pass := diff < 0.002 && diff > -0.002
  IO.println s!"fp16 v1 (0.1): {pass}"
  checks := pass :: checks

  -- IEEE754 -0.0 == 0.0
  let v2 <- IO.ofExcept (decode 4)
  let pass := v2 == 0.0
  IO.println s!"fp16 v2 (-0): {pass}"
  checks := pass :: checks

  -- +inf
  let v3 <- IO.ofExcept (decode 6)
  let pass := v3 == posInf
  IO.println s!"fp16 v3 (inf): {pass}"
  checks := pass :: checks

  -- -inf
  let v4 <- IO.ofExcept (decode 8)
  let pass := v4 == negInf
  IO.println s!"fp16 v4 (-inf): {pass}"
  checks := pass :: checks

  -- nan != nan
  let v5 <- IO.ofExcept (decode 10)
  let pass := v5 != v5
  IO.println s!"fp16 v5 (nan): {pass}"
  checks := pass :: checks

  -- 2048.5 rounds to 2048 in fp16 npy (step size is 2)
  let v6 <- IO.ofExcept (decode 12)
  IO.println s!"fp16 v6 actual: {v6}"
  let pass := v6 == Float32.ofNat 2048
  IO.println s!"fp16 v6 (2048.5 rounds to 2048): {pass}"
  checks := pass :: checks

  -- 100.5 fits in fp16
  let v7 <- IO.ofExcept (decode 14)
  let pass := v7 == 100.5
  IO.println s!"fp16 v7 (100.5) : {pass}"
  checks := pass :: checks

  -- subnormals rounding
  let v8 <- IO.ofExcept (decode 16)
  let diff8 := v8 - 0.00005
  let pass := diff8 < 0.000001 && diff8 > -0.000001
  IO.println s!"fp16 v8 (subnormal 0.00005): {pass}, actual: {v8}"
  checks := pass :: checks

  -- below fp16 minimum subnormal so should be 0
  let v9 <- IO.ofExcept (decode 18)
  let pass := v9 == 0.0
  IO.println s!"fp16 v9 (1.16e-10 -> 0): {pass}"
  checks := pass :: checks

  -- more values below fp16 min subnormal → zero
  let v10 <- IO.ofExcept (decode 20)
  let pass := v10 == 0.0
  IO.println s!"fp16 v10 (1e-20 -> 0): {pass}"
  checks := pass :: checks

  let v11 <- IO.ofExcept (decode 22)
  let pass := v11 == 0.0
  IO.println s!"fp16 v11 (-1e-20 -> -0): {pass}"
  checks := pass :: checks

  let v12 <- IO.ofExcept (decode 24)
  let pass := v12 == 0.0
  IO.println s!"fp16 v12 (1e-40 -> 0): {pass}"
  checks := pass :: checks

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
  checks := pass :: checks

  -- 0.1 rounded in bf16
  let v1 <- IO.ofExcept (decode 2)
  let diff := v1 - 0.1
  let pass := diff < 0.01 && diff > -0.01
  IO.println s!"bf16 v1 (0.1): {pass}"
  checks := pass :: checks

  -- -0
  let v2 <- IO.ofExcept (decode 4)
  let pass := v2 == 0.0
  IO.println s!"bf16 v2 (-0): {pass}"
  checks := pass :: checks

  -- +inf
  let v3 <- IO.ofExcept (decode 6)
  let pass := v3 == Float32.ofBits 0x7F800000
  IO.println s!"bf16 v3 (inf): {pass}"
  checks := pass :: checks

  -- -inf
  let v4 <- IO.ofExcept (decode 8)
  let pass := v4 == Float32.ofBits 0xFF800000
  IO.println s!"bf16 v4 (-inf): {pass}"
  checks := pass :: checks

  -- NaN
  let v5 <- IO.ofExcept (decode 10)
  let pass := v5 != v5
  IO.println s!"bf16 v5 (NaN): {pass}"
  checks := pass :: checks

  -- max bf16 value
  let v6 <- IO.ofExcept (decode 12)
  let pass := v6 == Float32.ofBits 0x7F7F0000
  IO.println s!"bf16 v6 (max): {pass}"
  checks := pass :: checks

  -- Arithmetic correctness: compare against ml_dtypes results
  let a := toLEByteArray (Float32.ofNat 3 / Float32.ofNat 2).toBFloat16Bits  -- bf16 encoding of 1.5
  let b := toLEByteArray (Float32.ofNat 5 / Float32.ofNat 2).toBFloat16Bits  -- bf16 encoding of 2.5

  let pass <- checkBits "bf16 add (1.5 + 2.5 = 4.0)" 16512 (Dtype.add .bfloat16 a b)
  checks := pass :: checks

  let pass <- checkBits "bf16 sub (1.5 - 2.5 = -1.0)" 49024 (Dtype.sub .bfloat16 a b)
  checks := pass :: checks

  let pass <- checkBits "bf16 mul (1.5 * 2.5 = 3.75)" 16496 (Dtype.mul .bfloat16 a b)
  checks := pass :: checks

  let pass <- checkBits "bf16 div (1.5 / 2.5 = 0.6)" 16154 (Dtype.div .bfloat16 a b)
  checks := pass :: checks

  -- abs(-1.5) = 1.5, ml_dtypes gives bits 16320
  let negA := toLEByteArray (Float32.ofNat 3 / Float32.ofNat 2).neg.toBFloat16Bits
  let pass <- checkBits "bf16 abs (-1.5) = 1.5" 16320 (Dtype.abs .bfloat16 negA)
  checks := pass :: checks

  -- Casting test cases
  -- bf16(42) -> float32 should give 42.0
  let bf42 := toLEByteArray (Float32.ofNat 42).toBFloat16Bits
  let castToF32 <- IO.ofExcept (Dtype.castOverflow .bfloat16 bf42 .float32)
  let f32Val <- IO.ofExcept (Float32.ofLEByteArray castToF32)
  let pass := f32Val == 42.0
  IO.println s!"bf16 cast bf16 to f32 (42): {pass}"
  checks := pass :: checks

  -- bf16(42) -> int8 should give 42
  let castToI8 <- IO.ofExcept (Dtype.castOverflow .bfloat16 bf42 .int8)
  let pass := castToI8.toNat == 42
  IO.println s!"bf16 cast bf16 to int8 (42): {pass}"
  checks := pass :: checks

  -- f32(3.14) -> bf16 should give bits 16457
  let f32_314 := toLEByteArray (Float32.ofNat 314 / Float32.ofNat 100)
  let pass <- checkBits "bf16 fp32 to bf16 (3.14)" 16457 (Dtype.castOverflow .float32 f32_314 .bfloat16)
  checks := pass :: checks

  -- float16(1.5) -> bfloat16 should give bits 16320
  let f16_15 := toLEByteArray (Float32.ofNat 3 / Float32.ofNat 2).toFloat16Bits
  let pass <- checkBits "bf16 fp16 to bf16 (1.5)" 16320 (Dtype.castOverflow .float16 f16_15 .bfloat16)
  checks := pass :: checks

  -- bf16(-0.0) -> bool should give 0 (false)
  let bf16NegZero := toLEByteArray (0x8000 : UInt16)
  let castToBool <- IO.ofExcept (Dtype.castOverflow .bfloat16 bf16NegZero .bool)
  let pass := castToBool == ByteArray.mk #[0]
  IO.println s!"bf16 -0 to bool (false): {pass}"
  checks := pass :: checks

  -- bf16(inf) to fp32 should give inf
  let bf16Inf := toLEByteArray (0x7F80 : UInt16)
  let castInfToF32 <- IO.ofExcept (Dtype.castOverflow .bfloat16 bf16Inf .float32)
  let infVal <- IO.ofExcept (Float32.ofLEByteArray castInfToF32)
  let pass := infVal == Float32.ofBits 0x7F800000
  IO.println s!"bf16 inf to fp32: {pass}"
  checks := pass :: checks

  -- Cast: bf16(-1.5) -> uint8 (NOTE: diverges from numpy which gives 255 via wrapping)
  -- Lean's Float32.toNat forces negatives to 0
  let castNegToU8 <- IO.ofExcept (Dtype.castOverflow .bfloat16 negA .uint8)
  let pass := castNegToU8.toNat == 0
  IO.println s!"bf16 cast -1.5 to uint8 (clamped to 0): {pass}"
  checks := pass :: checks

  -- Cast: bf16(inf) -> uint8 = 255 in both lean and numpy
  let castInfToU8 <- IO.ofExcept (Dtype.castOverflow .bfloat16 bf16Inf .uint8)
  let pass := castInfToU8.toNat == 255
  IO.println s!"bf16 cast inf to uint8 (255): {pass}"
  checks := pass :: checks

  return checks.all id

-- float8_e4m3fn edge cases: decode from npy, arithmetic, and casting
-- E4M3FN: 1 sign + 4 exponent + 3 mantissa, bias=7, max=448, no inf, only NaN
-- verified against ml_dtypes outputs
private def testFloat8E4M3EdgeCases : IO Bool := do
  let file <- saveNumpyArray "np.array([448.0, 0.1, -0.0, 0.001953125, 1.5, 0.5, 16.0, 256.0]).astype(__import__('ml_dtypes').float8_e4m3fn)"
  let npy <- Npy.parseFile file
  let arr <- IO.ofExcept (Tensor.ofNpy npy)
  let _ <- IO.FS.removeFile file
  -- decode 1-byte fp8 element at offset using extract
  let decode (offset : Nat) : Err Float32 := Dtype.decodeFloat8E4M3 (arr.data.extract offset (offset + 1))
  let mut checks : List Bool := []

  -- max representable value in e4m3fn (exp=15, mant=6 -> (1+6/8)*2^8 = 448)
  let v0 <- IO.ofExcept (decode 0)
  let pass := v0 == Float32.ofBits 0x43E00000
  IO.println s!"fp8_e4m3 v0 (448): {pass}"
  checks := pass :: checks

  -- 0.1 is not exactly representable; rounds to 0.1015625 in e4m3
  let v1 <- IO.ofExcept (decode 1)
  let diff := v1 - 0.1015625
  let pass := diff < 0.0001 && diff > -0.0001
  IO.println s!"fp8_e4m3 v1 (0.1 ~ 0.1015625): {pass}"
  checks := pass :: checks

  -- negative zero: IEEE 754 says -0.0 == 0.0
  let v2 <- IO.ofExcept (decode 2)
  let pass := v2 == 0.0
  IO.println s!"fp8_e4m3 v2 (-0): {pass}"
  checks := pass :: checks

  -- smallest subnormal: bits=1, value = 1 * 2^(-9) = 0.001953125
  let v3 <- IO.ofExcept (decode 3)
  let pass := v3 == Float32.ofBits 0x3B000000
  IO.println s!"fp8_e4m3 v3 (smallest subnormal): {pass}"
  checks := pass :: checks

  -- 1.5 is exactly representable (exp=7, mant=4 -> (1+4/8)*2^0 = 1.5)
  let v4 <- IO.ofExcept (decode 4)
  let pass := v4 == 1.5
  IO.println s!"fp8_e4m3 v4 (1.5): {pass}"
  checks := pass :: checks

  -- 0.5 is exactly representable (exp=6, mant=0 -> 1.0*2^(-1) = 0.5)
  let v5 <- IO.ofExcept (decode 5)
  let pass := v5 == 0.5
  IO.println s!"fp8_e4m3 v5 (0.5): {pass}"
  checks := pass :: checks

  -- 16 = 2^4 = maxSafeNat (largest consecutive integer)
  let v6 <- IO.ofExcept (decode 6)
  let pass := v6 == Float32.ofNat 16
  IO.println s!"fp8_e4m3 v6 (16): {pass}"
  checks := pass :: checks

  -- 256 = 2^8, exactly representable (exp=15, mant=0)
  let v7 <- IO.ofExcept (decode 7)
  let pass := v7 == Float32.ofNat 256
  IO.println s!"fp8_e4m3 v7 (256): {pass}"
  checks := pass :: checks

  -- Arithmetic: compare against ml_dtypes results
  let a := toLEByteArray (68 : UInt8)   -- e4m3 encoding of 3.0
  let b := toLEByteArray (52 : UInt8)   -- e4m3 encoding of 0.75
  let negA := toLEByteArray (196 : UInt8)  -- e4m3 encoding of -3.0

  let pass <- checkBitsU8 "fp8_e4m3 add (3.0 + 0.75 = 3.75)" 71 (Dtype.add .float8_e4m3 a b)
  checks := pass :: checks

  let pass <- checkBitsU8 "fp8_e4m3 sub (3.0 - 0.75 = 2.25)" 65 (Dtype.sub .float8_e4m3 a b)
  checks := pass :: checks

  let pass <- checkBitsU8 "fp8_e4m3 mul (3.0 * 0.75 = 2.25)" 65 (Dtype.mul .float8_e4m3 a b)
  checks := pass :: checks

  let pass <- checkBitsU8 "fp8_e4m3 div (3.0 / 0.75 = 4.0)" 72 (Dtype.div .float8_e4m3 a b)
  checks := pass :: checks

  let pass <- checkBitsU8 "fp8_e4m3 abs (-3.0) = 3.0" 68 (Dtype.abs .float8_e4m3 negA)
  checks := pass :: checks

  -- Casting: e4m3(3.0) -> int8 truncates to 3
  let castToI8 <- IO.ofExcept (Dtype.castOverflow .float8_e4m3 a .int8)
  let pass := castToI8.toNat == 3
  IO.println s!"fp8_e4m3 cast to int8 (3.0 -> 3): {pass}"
  checks := pass :: checks

  -- e4m3(-0.0) -> bool should be false (isZero handles -0 correctly)
  let negZero := toLEByteArray (128 : UInt8)
  let castToBool <- IO.ofExcept (Dtype.castOverflow .float8_e4m3 negZero .bool)
  let pass := castToBool == ByteArray.mk #[0]
  IO.println s!"fp8_e4m3 -0 to bool (false): {pass}"
  checks := pass :: checks

  -- fp32(3.0) -> e4m3 should give bits 68
  let f32_3 := toLEByteArray (Float32.ofNat 3)
  let castToE4m3 <- IO.ofExcept (Dtype.castOverflow .float32 f32_3 .float8_e4m3)
  let pass := castToE4m3 == toLEByteArray (68 : UInt8)
  IO.println s!"fp8_e4m3 fp32 to e4m3 (3.0): {pass}"
  checks := pass :: checks

  -- [448, 464] saturate to 448 (matches ml_dtypes)
  let f32_460 := toLEByteArray (Float32.ofNat 460)
  let castSaturate <- IO.ofExcept (Dtype.castOverflow .float32 f32_460 .float8_e4m3)
  let pass := castSaturate == toLEByteArray (126 : UInt8)  -- 448.0
  IO.println s!"fp8_e4m3 saturate (460 -> 448): {pass}"
  checks := pass :: checks

  -- Values >= 465 overflow to NaN (matches ml_dtypes)
  let f32_465 := toLEByteArray (Float32.ofNat 465)
  let castOverflow <- IO.ofExcept (Dtype.castOverflow .float32 f32_465 .float8_e4m3)
  let pass := castOverflow == toLEByteArray (127 : UInt8)  -- NaN
  IO.println s!"fp8_e4m3 overflow (465 -> NaN): {pass}"
  checks := pass :: checks

  return checks.all id

-- float8_e5m2 edge cases: decode from npy, arithmetic, and casting
-- E5M2: 1 sign + 5 exponent + 2 mantissa, bias=15, max=57344, has inf and NaN
-- verified against ml_dtypes outputs
private def testFloat8E5M2EdgeCases : IO Bool := do
  let file <- saveNumpyArray "np.array([57344.0, 0.1, -0.0, 1.52587890625e-05, 1.5, 0.5, 8.0, 2.0]).astype(__import__('ml_dtypes').float8_e5m2)"
  let npy <- Npy.parseFile file
  let arr <- IO.ofExcept (Tensor.ofNpy npy)
  let _ <- IO.FS.removeFile file
  -- Decode 1-byte e5m2 element at offset using extract
  let decode (offset : Nat) : Err Float32 :=
    Dtype.decodeFloat8E5M2 (arr.data.extract offset (offset + 1))
  let mut checks : List Bool := []

  -- max representable value (57344)
  let v0 <- IO.ofExcept (decode 0)
  let pass := v0 == Float32.ofBits 0x47600000
  IO.println s!"fp8_e5m2 v0 (57344): {pass}"
  checks := pass :: checks

  -- 0.1 rounded in e5m2
  let v1 <- IO.ofExcept (decode 1)
  let diff := v1 - 0.09375
  let pass := diff < 0.01 && diff > -0.01
  IO.println s!"fp8_e5m2 v1 (0.1 ~ 0.09375): {pass}"
  checks := pass :: checks

  -- -0
  let v2 <- IO.ofExcept (decode 2)
  let pass := v2 == 0.0
  IO.println s!"fp8_e5m2 v2 (-0): {pass}"
  checks := pass :: checks

  -- smallest subnormal: 2^(-16)
  let v3 <- IO.ofExcept (decode 3)
  let pass := v3 == Float32.ofBits 0x37800000
  IO.println s!"fp8_e5m2 v3 (smallest subnormal): {pass}"
  checks := pass :: checks

  -- 1.5
  let v4 <- IO.ofExcept (decode 4)
  let pass := v4 == 1.5
  IO.println s!"fp8_e5m2 v4 (1.5): {pass}"
  checks := pass :: checks

  -- 0.5
  let v5 <- IO.ofExcept (decode 5)
  let pass := v5 == 0.5
  IO.println s!"fp8_e5m2 v5 (0.5): {pass}"
  checks := pass :: checks

  -- 8 = maxSafeNat
  let v6 <- IO.ofExcept (decode 6)
  let pass := v6 == Float32.ofNat 8
  IO.println s!"fp8_e5m2 v6 (8): {pass}"
  checks := pass :: checks

  -- 2.0
  let v7 <- IO.ofExcept (decode 7)
  let pass := v7 == 2.0
  IO.println s!"fp8_e5m2 v7 (2.0): {pass}"
  checks := pass :: checks

  -- Arithmetic: 1.5 and 2.0
  let a := toLEByteArray (62 : UInt8)   -- e5m2 encoding of 1.5
  let b := toLEByteArray (64 : UInt8)   -- e5m2 encoding of 2.0
  let negA := toLEByteArray (190 : UInt8)  -- e5m2 encoding of -1.5

  let pass <- checkBitsU8 "fp8_e5m2 add (1.5 + 2.0 = 3.5)" 67 (Dtype.add .float8_e5m2 a b)
  checks := pass :: checks

  let pass <- checkBitsU8 "fp8_e5m2 sub (1.5 - 2.0 = -0.5)" 184 (Dtype.sub .float8_e5m2 a b)
  checks := pass :: checks

  let pass <- checkBitsU8 "fp8_e5m2 mul (1.5 * 2.0 = 3.0)" 66 (Dtype.mul .float8_e5m2 a b)
  checks := pass :: checks

  let pass <- checkBitsU8 "fp8_e5m2 div (1.5 / 2.0 = 0.75)" 58 (Dtype.div .float8_e5m2 a b)
  checks := pass :: checks

  let pass <- checkBitsU8 "fp8_e5m2 abs (-1.5) = 1.5" 62 (Dtype.abs .float8_e5m2 negA)
  checks := pass :: checks

  -- Casting: e5m2(1.5) -> int8 = 1
  let castToI8 <- IO.ofExcept (Dtype.castOverflow .float8_e5m2 a .int8)
  let pass := castToI8.toNat == 1
  IO.println s!"fp8_e5m2 cast to int8 (1.5 -> 1): {pass}"
  checks := pass :: checks

  -- e5m2(-0) -> bool = false
  let negZero := toLEByteArray (128 : UInt8)
  let castToBool <- IO.ofExcept (Dtype.castOverflow .float8_e5m2 negZero .bool)
  let pass := castToBool == ByteArray.mk #[0]
  IO.println s!"fp8_e5m2 -0 to bool (false): {pass}"
  checks := pass :: checks

  -- fp32(2.0) -> e5m2 = bits 64
  let f32_2 := toLEByteArray (Float32.ofNat 2)
  let castToE5m2 <- IO.ofExcept (Dtype.castOverflow .float32 f32_2 .float8_e5m2)
  let pass := castToE5m2 == toLEByteArray (64 : UInt8)
  IO.println s!"fp8_e5m2 fp32 to e5m2 (2.0): {pass}"
  checks := pass :: checks

  -- values > 57344 but < 65535 = max
  let f32_60000 := toLEByteArray (Float32.ofNat 60000)
  let castSaturate <- IO.ofExcept (Dtype.castOverflow .float32 f32_60000 .float8_e5m2)
  let pass := castSaturate == toLEByteArray (123 : UInt8)  -- 57344
  IO.println s!"fp8_e5m2 saturate (60000 -> 57344): {pass}"
  checks := pass :: checks

  -- Overflow to inf: >= 65535
  let f32_65536 := toLEByteArray (Float32.ofNat 65536)
  let castOverflow <- IO.ofExcept (Dtype.castOverflow .float32 f32_65536 .float8_e5m2)
  let pass := castOverflow == toLEByteArray (124 : UInt8)  -- +inf
  IO.println s!"fp8_e5m2 overflow (65536 -> inf): {pass}"
  checks := pass :: checks

  -- +inf preserved
  let f32_inf := toLEByteArray (Float32.ofBits 0x7F800000)
  let castInf <- IO.ofExcept (Dtype.castOverflow .float32 f32_inf .float8_e5m2)
  let pass := castInf == toLEByteArray (124 : UInt8)
  IO.println s!"fp8_e5m2 +inf -> +inf: {pass}"
  checks := pass :: checks

  -- -inf preserved
  let f32_negInf := toLEByteArray (Float32.ofBits 0xFF800000)
  let castNegInf <- IO.ofExcept (Dtype.castOverflow .float32 f32_negInf .float8_e5m2)
  let pass := castNegInf == toLEByteArray (252 : UInt8)
  IO.println s!"fp8_e5m2 -inf -> -inf: {pass}"
  checks := pass :: checks

  -- e5m2(+inf) -> int8 = -1 (INT64_MAX truncated to int8)
  let infBytes := toLEByteArray (124 : UInt8)  -- e5m2 +inf
  let castInfToI8 <- IO.ofExcept (Dtype.castOverflow .float8_e5m2 infBytes .int8)
  let pass := castInfToI8.toInt == 127
  IO.println s!"fp8_e5m2 +inf -> int8 (127): {pass}"
  checks := pass :: checks

  -- e5m2(-inf) -> int8 = -128
  let negInfBytes := toLEByteArray (252 : UInt8)  -- e5m2 -inf
  let castNegInfToI8 <- IO.ofExcept (Dtype.castOverflow .float8_e5m2 negInfBytes .int8)
  let pass := castNegInfToI8.toInt == -128
  IO.println s!"fp8_e5m2 -inf -> int8 (-128): {pass}"
  checks := pass :: checks

  -- e5m2(NaN) -> int8 = 0
  let nanBytes := toLEByteArray (126 : UInt8)  -- e5m2 NaN
  let castNanToI8 <- IO.ofExcept (Dtype.castOverflow .float8_e5m2 nanBytes .int8)
  let pass := castNanToI8.toInt == 0
  IO.println s!"fp8_e5m2 NaN -> int8 (0): {pass}"
  checks := pass :: checks

  -- e5m2 -> e4m3: +inf -> NaN (e4m3 has no inf)
  let castInfToE4m3 <- IO.ofExcept (Dtype.castOverflow .float8_e5m2 infBytes .float8_e4m3)
  let pass := castInfToE4m3 == toLEByteArray (127 : UInt8)  -- e4m3 NaN
  IO.println s!"fp8_e5m2 +inf -> e4m3 (NaN): {pass}"
  checks := pass :: checks

  -- e5m2(+inf) -> uint8 = 255 (UINT64_MAX truncated to uint8)
  let castInfToU8 <- IO.ofExcept (Dtype.castOverflow .float8_e5m2 infBytes .uint8)
  let pass := castInfToU8.toNat == 255
  IO.println s!"fp8_e5m2 +inf -> uint8 (255): {pass}"
  checks := pass :: checks

  -- e5m2(-inf) -> uint8 = 0
  let castNegInfToU8 <- IO.ofExcept (Dtype.castOverflow .float8_e5m2 negInfBytes .uint8)
  let pass := castNegInfToU8.toNat == 0
  IO.println s!"fp8_e5m2 -inf -> uint8 (0): {pass}"
  checks := pass :: checks

  -- e5m2(+inf) -> int32: we give 2147483647, numpy gives INT32_MAX
  let castInfToI32 <- IO.ofExcept (Dtype.castOverflow .float8_e5m2 infBytes .int32)
  let pass := castInfToI32.toInt == 2147483647
  IO.println s!"fp8_e5m2 +inf -> int32 (2147483647, numpy gives INT32_MAX): {pass}"
  checks := pass :: checks

  return checks.all id

def runAllTests : IO Bool := do
 return (<- testTensorElementBV Dtype.uint16) &&
        (<- testTensorElementBV Dtype.uint32) &&
        (<- testFloat16EdgeCases) &&
        (<- testBFloat16EdgeCases) &&
        (<- testFloat8E4M3EdgeCases) &&
        (<- testFloat8E5M2EdgeCases)

end Test
end TensorLib
