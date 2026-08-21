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
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Std.Tactic.BVDecide
import TensorLib.Dtype
import TensorLib.Float
import TensorLib.LOrd

-- Silence stylistic proof-hygiene linters (unused simp lemmas / redundant tactic
-- branches inside `first`/`try` combinators). These are cosmetic and do not affect
-- correctness; the genuine `declaration uses sorry` notices are intentionally kept.
set_option linter.unusedSimpArgs false
set_option linter.unusedTactic false
set_option linter.unreachableTactic false
set_option linter.unusedVariables false

namespace TensorLib

-- # of mantissa bits for each compute dtype
def mantissaBits (dtype : Dtype) : Option Nat := match dtype with
  | .float8_e4m3 => some 3
  | .float8_e5m2 => some 2
  | .float8_e3m4 => some 4
  | .float16 => some 10
  | .bfloat16 => some 7
  | .float32 => some 23
  | .float64 => some 52
  | _ => none

-- Machine epsilon: ε = 2^(-mantissa_bits).
-- The gap between adjacent representable values at magnitude 1.
def machineEpsilon (dtype : Dtype) : Option Float32 :=
  (mantissaBits dtype).map (fun k => Float32.pow 2.0 (-k.toFloat32))

-- v = x + (v - x) for Float32 — basic algebra, unprovable without Mathlib.
-- TODO: replace with a proof once Float32 algebraic instances are available.
axiom float32_eq_add_sub (v x : Float32) : v = x + (v - x)

-- When the fp32 exponent field is 0xFF, x is NaN or ±inf.
-- NaN and inf are not ≤ 448.0, so this contradicts hNoOverflow.
-- TODO: replace with a proof once Float32 bit-pattern lemmas are available.
axiom float32_exp_ff_not_finite (x : Float32) :
  (x.toBits >>> 23 &&& 0xFF = 0xFF) → ¬(x.abs <= 448.0)

-- When fp32 exponent field is 0 and x ≠ 0, x is a fp32 subnormal.
-- fp32 subnormals are smaller than the smallest E4M3 subnormal (2^(-9)).
-- TODO: replace with a proof once Float32 bit-pattern lemmas are available.
axiom float32_exp_zero_abs_small (x : Float32) :
    (x.toBits >>> 23 &&& 0xFF = 0) → x ≠ 0 → x.abs < Float32.ofBits 0x3B000000

-- `≤` and reverse-`<` are incompatible (proved from the IEEE compare model in LOrd).
theorem float32_le_lt_false (a b : Float32) : a <= b -> b < a -> False :=
  Float32.le_lt_false a b

-- Triangle inequality for Float32 absolute value.
-- |a + b| ≤ |a| + |b|. Holds for all non-NaN finite Float32 values.
-- NaN excluded: NaN.abs is NaN and NaN ≤ anything is false.
axiom float32_abs_triangle (a b : Float32)
    (ha : a.isNaN = false) (hb : b.isNaN = false) :
    (a + b).abs ≤ a.abs + b.abs

-- Monotonicity of ≤ for Float32 addition on non-NaN values.
-- If a ≤ b and c ≤ d then a + c <= b + d.
-- NaN excluded: NaN comparisons always return false.
axiom float32_add_le_add (a b c d : Float32)
    (ha : a.isNaN = false) (hb : b.isNaN = false)
    (hc : c.isNaN = false) (hd : d.isNaN = false) :
    a ≤ b → c ≤ d → a + c ≤ b + d

-- Transitivity of ≤ for non-NaN Float32 values.
-- NaN excluded: NaN ≤ anything is false, so transitivity fails for NaN.
theorem float32_le_trans (a b c : Float32)
    (ha : a.isNaN = false) (hb : b.isNaN = false) (hc : c.isNaN = false) :
    a ≤ b → b ≤ c → a ≤ c :=
  Float32.le_trans_of_not_nan a b c ha hb hc

-- Distributivity: k * (a + b + c) = k * a + k * b + k * c for non-NaN values.
axiom float32_mul_add3 (k a b c : Float32)
    (hk : k.isNaN = false) (ha : a.isNaN = false)
    (hb : b.isNaN = false) (hc : c.isNaN = false) :
    k * (a + b + c) = k * a + k * b + k * c

-- Commutativity of Float32 addition for non-NaN values: a + b = b + a.
-- NaN excluded: NaN + b = NaN and b + NaN = NaN, but NaN != NaN in IEEE 754.
axiom float32_add_comm (a b : Float32)
    (ha : a.isNaN = false) (hb : b.isNaN = false) :
    a + b = b + a

-- Add/sub cancellation for non-NaN Float32: (a + b) - b = a.
axiom float32_add_sub_cancel (a b : Float32)
    (ha : a.isNaN = false) (hb : b.isNaN = false) :
    a + b - b = a

-- Left cancellation: a + b - a = b (i.e., the first summand cancels).
axiom float32_add_sub_cancel_left (a b : Float32)
    (ha : a.isNaN = false) (hb : b.isNaN = false) :
    a + b - a = b

-- Subtraction decomposition for non-NaN values:
-- (a' + b') - (a + b) = (a' - a) + (b' - b).
-- True over reals; conditioned on non-NaN to avoid NaN propagation.
axiom float32_add_sub_decomp (a b a' b' : Float32)
    (ha : a.isNaN = false) (hb : b.isNaN = false)
    (ha' : a'.isNaN = false) (hb' : b'.isNaN = false) :
    (a' + b') - (a + b) = (a' - a) + (b' - b)

-- Sub-additivity of abs for subtraction on non-NaN values: |a - c| <= |a - b| + |b - c|.
axiom float32_abs_sub_triangle (a b c : Float32)
    (ha : a.isNaN = false) (hb : b.isNaN = false) (hc : c.isNaN = false) :
    (a - c).abs ≤ (a - b).abs + (b - c).abs

-- Float32 addition/subtraction preserve non-NaN for non-NaN inputs.
-- In IEEE 754, a + b is NaN only if a or b is NaN (ignoring inf-inf).
-- For finite non-NaN values this always holds.
axiom float32_add_notNaN (a b : Float32)
    (ha : a.isNaN = false) (hb : b.isNaN = false) :
    (a + b).isNaN = false

axiom float32_sub_notNaN (a b : Float32)
    (ha : a.isNaN = false) (hb : b.isNaN = false) :
    (a - b).isNaN = false

-- Helper: abs preserves isNaN for UnpackedFloat.
-- Direct case analysis — no Float32 machinery needed.
private lemma unpackedFloat_isNaN_abs_eq (u : Float.Model.UnpackedFloat) :
    u.abs.isNaN = u.isNaN := by
  cases u with
  | notANumber => simp [Float.Model.UnpackedFloat.abs, Float.Model.UnpackedFloat.isNaN]
  | infinity s | zero s => cases s <;> simp [Float.Model.UnpackedFloat.abs, Float.Model.UnpackedFloat.isNaN]
  | finite s m e hm => cases s <;> simp [Float.Model.UnpackedFloat.abs, Float.Model.UnpackedFloat.isNaN]

-- Float32.abs preserves non-NaN.
-- Proof: a.isNaN = a.toModel.unpack.isNaN (by definition),
-- and abs maps unpack to unpack.abs while preserving isNaN status
-- (notANumber→notANumber, infinity/zero/finite→positive variant).
-- The key step: (pack u.abs).unpack.isNaN = u.abs.isNaN, which for
-- notANumber/infinity/zero follows by rfl; for finite needs pack/unpack roundtrip.
-- Helper: (pack u).unpack.isNaN = u.isNaN for all UnpackedFloat u.
-- For notANumber/infinity/zero: rfl. For finite: pack never produces NaN bit pattern
-- (overflow → infinity, else → finite/zero with exponent ≠ 0xFF).
private lemma unpack_pack_isNaN (u : Float.Model.UnpackedFloat) :
    (Float32.Model.pack u).unpack.isNaN = u.isNaN := by
  cases u with
  | notANumber => rfl
  | infinity s | zero s => cases s <;> rfl
  | finite s m e hm =>
    -- (Float32.Model.pack f).unpack reduces (by rfl, via UInt32.toBitVec_ofBitVec) to
    -- UnpackedFloat.unpack binary32 (UnpackedFloat.pack binary32 f); and (finite ..).isNaN = false.
    show (Float.Model.UnpackedFloat.unpack Float.Model.Format.binary32
            (Float.Model.UnpackedFloat.pack Float.Model.Format.binary32
              (.finite s m e hm))).isNaN = false
    -- Only the `notANumber` constructor makes `isNaN` true; every other unpack result is
    -- non-NaN by `rfl`. So we case on the unpack result and refute the notANumber case.
    cases hu : Float.Model.UnpackedFloat.unpack Float.Model.Format.binary32
                 (Float.Model.UnpackedFloat.pack Float.Model.Format.binary32 (.finite s m e hm)) with
    | infinity _ => rfl
    | zero _ => rfl
    | finite _ _ _ _ => rfl
    | notANumber =>
      -- `unpack = notANumber` requires exponentVec = -1 (all ones); but pack of a finite
      -- value gives exponentVec = -1 only in the overflow branch, where mantissaVec = 0
      -- (yielding infinity, not NaN). So this case is impossible.
      exfalso
      revert hu
      simp only [Float.Model.UnpackedFloat.pack]
      split_ifs with hov hn <;>
      intro hu <;>
      simp only [Float.Model.UnpackedFloat.packedInfinity,
                 Float.Model.UnpackedFloat.unpack,
                 Float.Model.UnpackedFloat.unpackExponent_packComponents,
                 Float.Model.UnpackedFloat.unpackMantissa_packComponents] at hu <;>
      (try split_ifs at hu with he1 he2) <;>
      first
      | contradiction                      -- leaf ≠ notANumber (distinct ctors), or he1 : ¬True
      | (revert he2; decide)               -- overflow mantissa leaf: he2 : ¬(0 = 0#23) is false
      | (-- notANumber leaf: he1 : exponentVec = -1#8; refute (only overflow gives exp = -1)
         exfalso
         have htn := congr_arg BitVec.toNat he1
         simp only [Float.Model.Format.binary32, Float.Model.Format.exponentBias,
                    Float.Model.Format.mantissaBitsWithoutImplicit, Float.Model.Format.exponentBits,
                    BitVec.toNat_ofNat, BitVec.neg_one_eq_allOnes, BitVec.toNat_allOnes] at htn hov
         omega)

theorem float32_abs_notNaN (a : Float32) (ha : a.isNaN = false) : a.abs.isNaN = false := by
  -- a.abs.isNaN = a.abs.toModel.unpack.isNaN                (defeq)
  --             = (pack a.toModel.unpack.abs).unpack.isNaN  (a.abs.toModel = pack …, by rfl)
  --             = a.toModel.unpack.abs.isNaN                (unpack_pack_isNaN)
  --             = a.toModel.unpack.isNaN                    (unpackedFloat_isNaN_abs_eq)
  --             = false                                     (ha, defeq)
  show a.abs.toModel.unpack.isNaN = false
  rw [show a.abs.toModel = Float32.Model.pack a.toModel.unpack.abs from rfl,
      unpack_pack_isNaN, unpackedFloat_isNaN_abs_eq]
  exact ha

-- Float32 multiplication preserves non-NaN for non-NaN inputs.
axiom float32_mul_notNaN (a b : Float32)
    (ha : a.isNaN = false) (hb : b.isNaN = false) : (a * b).isNaN = false

-- Reflexivity of ≤ for non-NaN Float32 (NaN ≤ NaN is false in IEEE 754).
theorem float32_le_refl_notNaN (a : Float32) (ha : a.isNaN = false) : a ≤ a :=
  Float32.le_refl_of_not_nan a ha

-- If a ≤ b holds and b is non-NaN, then a is non-NaN.
-- Proof: by contrapositive. If a.isNaN = true then a.toModel.unpack = notANumber,
-- so notANumber.compare b.toModel.unpack = none, Option.any isLE none = false,
-- meaning a ≤ b = false — contradicting the hypothesis.
set_option maxHeartbeats 400000 in
theorem float32_le_left_notNaN (a b : Float32) (hb : b.isNaN = false) :
    a ≤ b → a.isNaN = false := by
  intro h
  by_contra ha
  simp only [ne_eq, Bool.not_eq_false] at ha
  -- ha : a.isNaN = true → a.toModel.unpack = notANumber
  have h_u : a.toModel.unpack = .notANumber := by
    cases h_uu : a.toModel.unpack with
    | notANumber => rfl
    | infinity s | zero s | finite s m e hm =>
      simp [Float32.isNaN, Float32.Model.isNaN, Float.Model.UnpackedFloat.isNaN, h_uu] at ha
  -- a ≤ b is false when a.toModel.unpack = notANumber
  -- use simp_all to reduce to h : false = true, then close by decide
  simp_all only [LE.le, instLEFloat32, Float32.le, Float32.Model.le,
                 Float.Model.UnpackedFloat.le, Float.Model.UnpackedFloat.compare,
                 Option.any, decide_eq_true_eq, h_u]
  exact absurd h (by decide)

-- abs does not change NaN status: a.abs.isNaN = false implies a.isNaN = false.
-- Proof: by contrapositive. If a.isNaN = true then a.toModel.unpack = notANumber.
-- Then a.abs.toModel.unpack = (pack notANumber).unpack = notANumber (by rfl),
-- so a.abs.isNaN = true, contradicting the hypothesis.
theorem float32_notNaN_of_abs_notNaN (a : Float32) : a.abs.isNaN = false → a.isNaN = false := by
  intro h
  by_contra ha
  -- ha : a.isNaN ≠ false, i.e., a.isNaN = true
  simp only [ne_eq, Bool.not_eq_false] at ha
  -- show a.abs.isNaN = true to contradict h
  have h_abs_NaN : a.abs.isNaN = true := by
    -- a.isNaN = true means a.toModel.unpack = notANumber
    -- abs of notANumber = notANumber, and (pack notANumber).unpack = notANumber by rfl
    simp only [Float32.isNaN, Float32.Model.isNaN, Float.Model.UnpackedFloat.isNaN] at ha ⊢
    -- ha says a.toModel.unpack.isNaN = true, i.e., = notANumber
    have h_u : a.toModel.unpack = .notANumber := by
      cases h_uu : a.toModel.unpack with
      | notANumber => rfl
      | infinity s | zero s | finite s m e hm =>
        simp [Float.Model.UnpackedFloat.isNaN, h_uu] at ha
    -- a.abs.toModel.unpack = (pack notANumber).unpack = notANumber
    have h_abs_u : a.abs.toModel.unpack = .notANumber := by
      -- rewrite step by step to avoid simp expanding a.toModel.toBits
      have heq : a.toModel.abs = Float32.Model.pack .notANumber := by
        unfold Float32.Model.abs
        rw [h_u]; simp [Float.Model.UnpackedFloat.abs]
      rw [show a.abs.toModel = a.toModel.abs from rfl, heq]
      rfl  -- (pack notANumber).unpack = notANumber by definitional reduction
    simp [Float32.isNaN, Float32.Model.isNaN, Float.Model.UnpackedFloat.isNaN, h_abs_u]
  simp [h_abs_NaN] at h

-- For a normal fp32 value (exp != 0, exp != 0xFF) in E4M3's range,
-- rounding introduces at most (1/2) * 2^(-3) * |x| error.
-- Proof requires bit-level case analysis on toFloat8E4M3Bits:
-- for E4M3 exponent e, adjacent values differ by 2^(e-10),
-- so error <= 2^(e-11) <= (1/2) * 2^(-3) * |x| since |x| >= 2^(e-7).
axiom float8E4M3_normal_rounding_bound (x : Float32)
    (hexp  : Not (x.toBits >>> 23 &&& 0xFF = 0xFF))
    (hexp0 : Not (x.toBits >>> 23 &&& 0xFF = 0))
    (hNoOverflow : x.abs <= 448.0) :
    (x.toFloat8E4M3Bits.toFloat32FromFloat8E4M3 - x).abs <=
    0.5 * Float32.pow 2.0 (-Nat.toFloat32 3) * x.abs

-- pointwise quantization error bound for fp8_e4m3
-- Rounding a fp32 value x to the nearest fp8_e4m3 value introduces
-- at most (1/2) * ε * |x| error, where ε = 2^(-3) = 0.125.
theorem pointwiseBoundE4M3 (x : Float32) (hNoOverflow : x.abs ≤ (Dtype.fp8Max .float8_e4m3).getD 0) (hNoUnderflow : x = 0 ∨ Float32.ofBits 0x3B000000 <= x.abs):
    ∃ err : Float32, Dtype.roundToComputeDtype x .float8_e4m3 = .ok (x + err)
    ∧ err.abs <= 0.5 * (machineEpsilon .float8_e4m3).getD 0 * x.abs := by
  simp [Dtype.roundToComputeDtype]
  unfold Dtype.decodeFloat8E4M3
  simp [ByteArray.size]
  -- provide the witness: the rounding error is the difference between rounded and original
  apply Exists.intro (x.toFloat8E4M3Bits.toFloat32FromFloat8E4M3 - x)
  apply And.intro
  · exact float32_eq_add_sub _ _
  · simp [machineEpsilon, mantissaBits]
    -- case 1: x = 0, round-trip error is 0, bound holds trivially
    by_cases hx : x = 0
    · subst hx
      native_decide
    · set exp := (x.toBits >>> 23) &&& 0xFF
      -- case 2: x is NaN or ±inf (exp = 0xFF), contradicts hNoOverflow
      by_cases hexp : exp = 0xFF
      · exact absurd (by simpa [Dtype.fp8Max] using hNoOverflow) (float32_exp_ff_not_finite x hexp)
      · -- case 3: x is a fp32 subnormal (exp = 0, x ≠ 0)
        -- contradicts hNoUnderflow since fp32 subnormals are below E4M3's smallest value
        by_cases hexp0 : exp = 0
        · exfalso
          cases hNoUnderflow with
          | inl h => exact hx h
          | inr h =>
            have hsmall := float32_exp_zero_abs_small x hexp0 hx
            exact float32_le_lt_false _ _ h hsmall
        · -- case 4: x is a normal fp32 value in E4M3's representable range
          -- exp != 0xFF (not NaN/inf), exp != 0 (not subnormal)
          -- so x is a finite normal fp32 value with |x| in [2^(-9), 448.0]
          exact float8E4M3_normal_rounding_bound x hexp hexp0
            (by simpa [Dtype.fp8Max] using hNoOverflow)

-- Generic axioms for all fp8 compute dtypes.
-- These replace the e4m3-specific axioms above and work for any dtype
-- where fp8Max is defined (i.e., float8_e4m3, float8_e5m2, float8_e3m4).

-- exp = 0xFF means NaN or ±inf, which is not ≤ any finite fp8 max.
-- Proof: exp = 0xFF → x.toModel.unpack = infinity _ or notANumber.
-- For infinity: infinity.le (finite v) = false.
-- For NaN: NaN.le anything = false.
-- Both follow from Float.Model.UnpackedFloat.compare semantics.
-- TODO: extract from Float32.toModel.unpack definition in Lean 4.33.
-- Helper: when exp = 0xFF, x.abs.toModel.unpack is NaN or +infinity.
-- We show this from the unpack definition: exponentVec = -1#_ → infinity or notANumber.
-- Then use abs semantics: abs maps both to notANumber or infinity .positive.
-- Helper: x.toModel.unpack is notANumber or infinity when exp field = 0xFF.
-- Proof: unpackExponent = -1#8 (all 1s) forces the infinity/NaN branch of unpack.
-- The bit manipulation uses UInt32.toBitVec_shiftRight/and to convert hexp to BitVec,
-- then bv_omega closes the extractLsb = -1#8 goal.
private lemma toModel_unpack_of_exp_ff (x : Float32)
    (hexp : x.toBits >>> 23 &&& 0xFF = 0xFF) :
    x.toModel.unpack = .notANumber ∨ ∃ s, x.toModel.unpack = .infinity s := by
  -- exp = 0xFF → unpackExponent (bits 30..23) = -1#8 (all 1s)
  -- TODO: the UInt32 → BitVec.extractLsb connection is:
  --   x.toBits >>> 23 &&& 0xFF = 0xFF (UInt32)
  --   ↔ x.toModel.toBits.toBitVec.extractLsb 30 23 = -1#8 (BitVec 8)
  -- This equivalence needs a lemma like UInt32.extractLsb_eq or omega after BitVec unfolding.
  have h_exp : x.toModel.toBits.toBitVec.extractLsb 30 23 = -1#8 := by
    -- Push the UInt32 hypothesis down to a BitVec equation, then let bv_decide
    -- discharge the extractLsb claim (bits 23..30 of the 32-bit pattern are all ones).
    have hbv : (x.toBits.toBitVec >>> 23) &&& 0xFF#32 = 0xFF#32 := by
      simpa using congrArg UInt32.toBitVec hexp
    show x.toBits.toBitVec.extractLsb 30 23 = -1#8
    bv_decide
  -- Case split on the actual unpack result; zero/finite are impossible since exp = 0xFF
  cases h_u : x.toModel.unpack with
  | notANumber => left; rfl
  | infinity s => right; exact ⟨s, rfl⟩
  | zero _ | finite _ _ _ _ =>
    -- unpackExponent = -1#8 forces infinity/NaN branch, contradicting zero/finite
    exfalso
    simp only [Float32.Model.unpack, Float.Model.UnpackedFloat.unpack,
               Float.Model.UnpackedFloat.unpackExponent, Float.Model.Format.binary32,
               Float32.toBits, h_exp, BitVec.cast_eq, ite_true] at h_u
    split_ifs at h_u <;> simp at h_u

-- Helper: when exp = 0xFF, x.abs.toModel.unpack is notANumber or infinity .positive.
-- abs maps notANumber → notANumber and infinity _ → infinity .positive.
-- The pack/unpack roundtrip holds by rfl (definitional equality in Lean 4.33).
private lemma abs_unpack_of_exp_ff (x : Float32)
    (hexp : x.toBits >>> 23 &&& 0xFF = 0xFF) :
    x.abs.toModel.unpack = .notANumber ∨
    x.abs.toModel.unpack = .infinity .positive := by
  -- Float32 is a structure: (ofModel m).toModel = m, so x.abs.toModel = x.toModel.abs
  have h_abs : x.abs.toModel = x.toModel.abs := rfl
  rcases toModel_unpack_of_exp_ff x hexp with h_u | ⟨s, h_u⟩
  · -- notANumber.abs = notANumber; (pack notANumber).unpack = notANumber by rfl
    left
    show (x.toModel.abs).unpack = .notANumber
    rw [show x.toModel.abs = Float32.Model.pack .notANumber by
      simp [Float32.Model.abs, h_u, Float.Model.UnpackedFloat.abs]]
    rfl
  · -- (infinity s).abs = infinity .positive; (pack (infinity .positive)).unpack = it by rfl
    right
    show (x.toModel.abs).unpack = .infinity .positive
    rw [show x.toModel.abs = Float32.Model.pack (.infinity .positive) by
      simp [Float32.Model.abs, h_u, Float.Model.UnpackedFloat.abs]]
    rfl

theorem fp8_exp_ff_not_finite (dtype : Dtype) (x : Float32)
    (hFp8 : (Dtype.fp8Max dtype).isSome)
    (hexp : x.toBits >>> 23 &&& 0xFF = 0xFF) :
    Not (x.abs ≤ (Dtype.fp8Max dtype).getD 0) := by
  rcases dtype with _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _
  all_goals simp [Dtype.fp8Max] at hFp8 ⊢
  -- For each fp8 dtype, show x.abs ≤ threshold is false
  all_goals (
    intro h
    -- x.abs is NaN or +infinity (from exp = 0xFF)
    -- unfold ≤ and substitute to get a false=true contradiction
    rcases abs_unpack_of_exp_ff x hexp with h_u | h_u
    · -- notANumber.compare _ = none → Option.any isLE none = false
      simp only [LE.le, instLEFloat32, Float32.le, decide_eq_true_eq,
                 Float32.Model.le, Float.Model.UnpackedFloat.le,
                 Float.Model.UnpackedFloat.compare, Option.any, h_u] at h
      exact absurd h (by decide)
    · -- (infinity .positive).compare (finite v) = some .gt → isLE = false
      simp only [LE.le, instLEFloat32, Float32.le, decide_eq_true_eq,
                 Float32.Model.le, Float.Model.UnpackedFloat.le,
                 Float.Model.UnpackedFloat.compare, Option.any, Ordering.isLE, h_u] at h
      exact absurd h (by decide))

-- pack/unpack round-trips on a fp32 subnormal: packing `finite positive m (-149)`
-- (with 0 < m < 2^23) takes the subnormal branch of `pack` and unpacks back unchanged.
private lemma subnormal_unpack_roundtrip (m : Nat) (hm : 0 < m) (hlt : m < 2 ^ 23) :
    (Float32.Model.pack (.finite .positive m (-149) hm)).unpack
      = .finite .positive m (-149) hm := by
  have hlog : m.log2 < 23 := (Nat.log2_lt (by omega)).mpr hlt
  have hmod : m % 2 ^ 23 = m := Nat.mod_eq_of_lt hlt
  have hmv : BitVec.ofNat 23 m ≠ 0#23 := by
    rw [Ne, ← BitVec.toNat_inj]
    simp only [BitVec.toNat_ofNat, BitVec.toNat_zero, hmod]; omega
  simp only [Float32.Model.pack, Float32.Model.unpack, UInt32.toBitVec_ofBitVec,
             Float.Model.UnpackedFloat.pack, Float.Model.Format.binary32,
             Float.Model.Format.exponentBias, Float.Model.Format.mantissaBits]
  -- Resolve `pack`'s branches: not overflow (biased exponent is 1), not normal (m.log2+1 ≤ 23).
  have hnm : ¬ (m.log2 + 1 = 1 + 23) := by omega
  -- Full `simp` evaluates the closed overflow condition, uses hnm for the normal branch,
  -- applies the packComponents unpack lemmas, and reduces the mantissa via hmod.
  simp [Float.Model.UnpackedFloat.unpack,
        Float.Model.UnpackedFloat.unpackExponent_packComponents,
        Float.Model.UnpackedFloat.unpackMantissa_packComponents,
        hnm, dif_neg hmv, hmod]
  -- Remaining: sign round-trips to positive, m < 2^23, and the exponent is -149.
  refine ⟨?_, ?_, ?_⟩
  · -- the sign bit of the packed value is 0 (independent of the mantissa)
    have hsign : Float.Model.UnpackedFloat.unpackSign
        (Float.Model.UnpackedFloat.packComponents Float.Model.Format.binary32
          .positive 0#8 (BitVec.ofNat 23 m)) = 0#1 := by
      unfold Float.Model.UnpackedFloat.unpackSign Float.Model.UnpackedFloat.packComponents
      simp only [Float.Model.Format.binary32, Float.Model.UnpackedFloat.Sign.toBitVec]
      bv_decide
    simp [Float.Model.UnpackedFloat.Sign.ofBitVec, hsign]
  · omega
  · simp only [Float.Model.Format.binary32, Float.Model.Format.exponentBias]; decide

-- When the fp32 exponent field is 0, `x` unpacks to either a (signed) zero or a
-- subnormal finite value with model exponent -149 and mantissa in [1, 2^23).
private lemma unpack_of_exp_zero (x : Float32) (hexp : x.toBits >>> 23 &&& 0xFF = 0) :
    (∃ s, x.toModel.unpack = .zero s) ∨
    (∃ s m, ∃ hm : 0 < m, m < 2 ^ 23 ∧ x.toModel.unpack = .finite s m (-149) hm) := by
  -- exponent field 0 → the unpacked exponent bitvector is 0#8
  have hev : Float.Model.UnpackedFloat.unpackExponent (spec := Float.Model.Format.binary32)
               x.toModel.toBits.toBitVec = 0#8 := by
    have hbv : (x.toBits.toBitVec >>> 23) &&& 0xFF#32 = 0#32 := by
      simpa using congrArg UInt32.toBitVec hexp
    unfold Float.Model.UnpackedFloat.unpackExponent
    simp only [Float.Model.Format.binary32]
    show ((x.toBits.toBitVec.extractLsb 30 23).cast _ : BitVec 8) = 0#8
    bv_decide
  -- the mantissa bitvector has toNat < 2^23 (it is a BitVec 23)
  set mv := Float.Model.UnpackedFloat.unpackMantissa (spec := Float.Model.Format.binary32)
              x.toModel.toBits.toBitVec with hmvdef
  by_cases hmant : mv = 0#23
  · left
    refine ⟨Float.Model.UnpackedFloat.Sign.ofBitVec
        (Float.Model.UnpackedFloat.unpackSign (spec := Float.Model.Format.binary32)
          x.toModel.toBits.toBitVec), ?_⟩
    show Float.Model.UnpackedFloat.unpack Float.Model.Format.binary32 x.toModel.toBits.toBitVec = _
    unfold Float.Model.UnpackedFloat.unpack
    simp only [Float.Model.Format.binary32, ← hmvdef, hev, hmant,
               show ((0#8 : BitVec 8) = -1#8) = False from by decide, if_false,
               show ((0#8 : BitVec 8) = 0#8) = True from by decide, if_true, dif_pos]
  · right
    have hpos : 0 < mv.toNat :=
      Nat.pos_of_ne_zero (fun h => hmant (BitVec.toNat_inj.mp (by simpa using h)))
    refine ⟨Float.Model.UnpackedFloat.Sign.ofBitVec
        (Float.Model.UnpackedFloat.unpackSign (spec := Float.Model.Format.binary32)
          x.toModel.toBits.toBitVec), mv.toNat, hpos, mv.isLt, ?_⟩
    show Float.Model.UnpackedFloat.unpack Float.Model.Format.binary32 x.toModel.toBits.toBitVec = _
    unfold Float.Model.UnpackedFloat.unpack
    simp only [Float.Model.Format.binary32, Float.Model.Format.exponentBias, ← hmvdef, hev,
               show ((0#8 : BitVec 8) = -1#8) = False from by decide, if_false,
               show ((0#8 : BitVec 8) = 0#8) = True from by decide, if_true, dif_neg hmant]
    congr 1

-- A +0 or a positive subnormal (model exponent -149) compares strictly below any
-- positive finite value whose model exponent E exceeds -149.
private lemma unpack_lt_fp8min (u : Float.Model.UnpackedFloat) (M : Nat) (E : Int) (hM : 0 < M)
    (hE : (-149 : Int) < E)
    (hu : u = .zero .positive ∨ ∃ m, ∃ hm : 0 < m, u = .finite .positive m (-149) hm) :
    (u.compare (.finite .positive M E hM) == some Ordering.lt) = true := by
  rcases hu with h | ⟨m, hm, h⟩ <;> subst h <;>
    simp [Float.Model.UnpackedFloat.compare, Int.compare_eq_lt.mpr hE, Ordering.then]

-- fp32 subnormals have value < 2^(-126), which is smaller than the minimum
-- of any fp8 format (smallest is 2^(-16) for e5m2).
theorem fp8_exp_zero_abs_small (dtype : Dtype) (x : Float32)
    (hFp8 : (Dtype.fp8Max dtype).isSome)
    (hexp : x.toBits >>> 23 &&& 0xFF = 0) (hx : x ≠ 0) :
    x.abs < (Dtype.fp8Min dtype).getD 0 := by
  -- |x| unpacks to +0 or a positive subnormal (model exponent -149), via abs + the roundtrip.
  have habs : x.abs.toModel.unpack = .zero .positive ∨
      ∃ m, ∃ hm : 0 < m, x.abs.toModel.unpack = .finite .positive m (-149) hm := by
    rcases unpack_of_exp_zero x hexp with ⟨s, hz⟩ | ⟨s, m, hm, hlt, hf⟩
    · left
      show (Float32.Model.pack x.toModel.unpack.abs).unpack = _
      rw [hz]; simp only [Float.Model.UnpackedFloat.abs]; rfl
    · right
      refine ⟨m, hm, ?_⟩
      show (Float32.Model.pack x.toModel.unpack.abs).unpack = _
      rw [hf]; simp only [Float.Model.UnpackedFloat.abs]
      exact subnormal_unpack_roundtrip m hm hlt
  -- Reduce `<` to a comparison on the unpacked values.
  show Float32.lt x.abs ((Dtype.fp8Min dtype).getD 0) = true
  unfold Float32.lt
  simp only [decide_eq_true_eq]
  show Float32.Model.lt x.abs.toModel ((Dtype.fp8Min dtype).getD 0).toModel = true
  unfold Float32.Model.lt Float.Model.UnpackedFloat.lt
  -- Only the three fp8 compute dtypes have `fp8Max`; discharge the rest via hFp8.
  rcases dtype with _|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_ <;>
    simp only [Dtype.fp8Max] at hFp8 <;>
    (try exact absurd hFp8 (by decide))
  -- Three fp8 dtypes remain; each fp8Min unpacks to a positive finite with exponent > -149.
  · -- float8_e4m3: 2^(-9), model exponent -32
    simp only [Dtype.fp8Min, Option.getD]
    rw [show (Float32.ofBits 0x3B000000).toModel.unpack
          = .finite .positive 8388608 (-32) (by norm_num) from rfl]
    exact unpack_lt_fp8min _ _ _ _ (by decide) habs
  · -- float8_e3m4: 2^(-6), model exponent -29
    simp only [Dtype.fp8Min, Option.getD]
    rw [show (Float32.ofBits 0x3C800000).toModel.unpack
          = .finite .positive 8388608 (-29) (by norm_num) from rfl]
    exact unpack_lt_fp8min _ _ _ _ (by decide) habs
  · -- float8_e5m2: 2^(-16), model exponent -39
    simp only [Dtype.fp8Min, Option.getD]
    rw [show (Float32.ofBits 0x37800000).toModel.unpack
          = .finite .positive 8388608 (-39) (by norm_num) from rfl]
    exact unpack_lt_fp8min _ _ _ _ (by decide) habs

-- For a normal fp32 value in any fp8 format's representable range,
-- rounding introduces at most (1/2) * machineEpsilon * |x| error.
-- Proof requires bit-level case analysis on the encode/decode round-trip
-- for each fp8 format. The key: for normal values at exponent e,
-- adjacent fp8 values differ by 2^(e - mantissaBits), so rounding
-- error ≤ half that gap ≤ (1/2) * 2^(-mantissaBits) * |x|.
-- TODO: requires format-specific bit-level reasoning.
axiom fp8_normal_rounding_bound (dtype : Dtype) (x : Float32)
    (hexp  : Not (x.toBits >>> 23 &&& 0xFF = 0xFF))
    (hexp0 : Not (x.toBits >>> 23 &&& 0xFF = 0))
    (hNoOverflow : x.abs ≤ (Dtype.fp8Max dtype).getD 0) :
    ∃ err : Float32, Dtype.roundToComputeDtype x dtype = .ok (x + err) ∧
    err.abs ≤ 0.5 * (machineEpsilon dtype).getD 0 * x.abs

-- Generic pointwise quantization error bound for all fp8 compute dtypes.
-- For any x that fits in dtype's representable range, rounding x to dtype
-- introduces at most (1/2) * epsilon * |x| absolute error.
-- hFp8 restricts to dtypes where fp8Max is defined (the three fp8 types).
theorem pointwiseBound (dtype : Dtype) (x : Float32)
    (hFp8 : (Dtype.fp8Max dtype).isSome)
    (hNoOverflow  : x.abs ≤ (Dtype.fp8Max dtype).getD 0)
    (hNoUnderflow : x = 0 ∨ (Dtype.fp8Min dtype).getD 0 ≤ x.abs) :
    ∃ err : Float32, Dtype.roundToComputeDtype x dtype = .ok (x + err) ∧
    err.abs ≤ 0.5 * (machineEpsilon dtype).getD 0 * x.abs := by
  by_cases hx : x = 0
  · -- x = 0: rounding error is 0, bound holds trivially
    -- provide witness err = 0 explicitly, then use native_decide for each fp8 dtype
    subst hx
    refine ⟨0, ?_, ?_⟩
    · -- roundToComputeDtype 0 dtype = .ok (0 + 0) = .ok 0
      rcases dtype with _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _
      all_goals (simp [Dtype.fp8Max] at hFp8; try native_decide)
    · -- (0 : Float32).abs = 0, so bound 0 ≤ 0.5 * epsilon * 0 holds
      rcases dtype with _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _
      all_goals (simp [Dtype.fp8Max] at hFp8; try native_decide)
  · set exp := (x.toBits >>> 23) &&& 0xFF
    -- case 2: x is NaN or ±inf, contradicts hNoOverflow
    by_cases hexp : exp = 0xFF
    · exact absurd (by simpa [Dtype.fp8Max] using hNoOverflow)
                   (fp8_exp_ff_not_finite dtype x hFp8 hexp)
    · -- case 3: x is a fp32 subnormal, contradicts hNoUnderflow
      by_cases hexp0 : exp = 0
      · exfalso
        cases hNoUnderflow with
        | inl h => exact hx h
        | inr h =>
          exact float32_le_lt_false _ _ h
            (fp8_exp_zero_abs_small dtype x hFp8 hexp0 hx)
      · -- case 4: x is a normal fp32 in dtype's range, use generic axiom
        exact fp8_normal_rounding_bound dtype x hexp hexp0
          (by simpa [Dtype.fp8Max] using hNoOverflow)

-- Higham-style addition error bound for fp8 dtypes.
-- Given a, b and their fp8 approximations a', b' with individual
-- quantization errors bounded by (u/2)|a| and (u/2)|b|, computing
-- roundToComputeDtype on the Float32 sum a' + b' gives a result within
-- (u/2)(|a| + |b| + |a'+b'|) of the true sum a + b.
-- where u = machineEpsilon dtype.
-- Proof structure:
--   result - (a+b) = err + (a'-a) + (b'-b)   [algebra]
--   |result - (a+b)| ≤ |err| + |a'-a| + |b'-b|   [triangle inequality]
--   ≤ (u/2)|a'+b'| + (u/2)|a| + (u/2)|b|   [pointwiseBound + ha' + hb']
--   = (u/2)(|a| + |b| + |a'+b'|)   [distributivity]
set_option maxHeartbeats 800000 in
theorem additionErrorBound (dtype : Dtype) (hFp8 : (Dtype.fp8Max dtype).isSome)
    (a b a' b' : Float32)
    -- non-NaN conditions: needed for the order and arithmetic axioms to hold
    (ha_nn : a.isNaN = false) (hb_nn : b.isNaN = false)
    (ha'_nn : a'.isNaN = false) (hb'_nn : b'.isNaN = false)
    -- a', b' are fp8 approximations of a, b with pointwise quantization error
    (ha' : (a' - a).abs ≤ 0.5 * (machineEpsilon dtype).getD 0 * a.abs)
    (hb' : (b' - b).abs ≤ 0.5 * (machineEpsilon dtype).getD 0 * b.abs)
    -- the sum a' + b' fits in dtype's representable range
    (hNoOverflow : (a' + b').abs ≤ (Dtype.fp8Max dtype).getD 0)
    (hNoUnderflow : a' + b' = 0 ∨
                    (Dtype.fp8Min dtype).getD 0 ≤ (a' + b').abs) :
    ∃ result : Float32,
      Dtype.roundToComputeDtype (a' + b') dtype = .ok result ∧
      (result - (a + b)).abs ≤
        0.5 * (machineEpsilon dtype).getD 0 * (a.abs + b.abs + (a' + b').abs) := by
  -- apply pointwiseBound to get the re-encoding error
  obtain ⟨err, hresult, herr⟩ := pointwiseBound dtype (a' + b') hFp8 hNoOverflow hNoUnderflow
  -- result = (a' + b') + err, so result - (a+b) = (a'-a) + (b'-b) + err
  refine ⟨a' + b' + err, hresult, ?_⟩
  -- Get non-NaN witnesses
  have hab'_nn  : (a' + b').isNaN = false  := float32_add_notNaN _ _ ha'_nn hb'_nn
  have hab_nn   : (a + b).isNaN = false    := float32_add_notNaN _ _ ha_nn hb_nn
  have ha'a_nn  : (a' - a).isNaN = false   := float32_sub_notNaN _ _ ha'_nn ha_nn
  have hb'b_nn  : (b' - b).isNaN = false   := float32_sub_notNaN _ _ hb'_nn hb_nn
  have hab'_abs_pre : (a' + b').abs.isNaN = false := float32_abs_notNaN _ hab'_nn
  have hu_nn_pre : (0.5 * (machineEpsilon dtype).getD 0).isNaN = false := by
    rcases dtype with _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _
    all_goals (simp [Dtype.fp8Max] at hFp8; try native_decide)
  have hu_rhs_nn : (0.5 * (machineEpsilon dtype).getD 0 * (a' + b').abs).isNaN = false :=
    float32_mul_notNaN _ _ hu_nn_pre hab'_abs_pre
  have herr_abs_nn : err.abs.isNaN = false :=
    float32_le_left_notNaN _ _ hu_rhs_nn herr
  have herr_nn  : err.isNaN = false :=
    float32_notNaN_of_abs_notNaN _ herr_abs_nn
  have hres_nn  : (a' + b' + err).isNaN = false := float32_add_notNaN _ _ hab'_nn herr_nn
  have ha'a_abs := float32_abs_notNaN _ ha'a_nn
  have hb'b_abs := float32_abs_notNaN _ hb'b_nn
  have ha_abs   := float32_abs_notNaN _ ha_nn
  have hb_abs   := float32_abs_notNaN _ hb_nn
  have hab'_abs := float32_abs_notNaN _ hab'_nn
  have hu_nn := hu_nn_pre
  -- |result - (a+b)| ≤ |result - (a'+b')| + |(a'+b') - (a+b)|
  have htri := float32_abs_sub_triangle (a' + b' + err) (a' + b') (a + b)
                 hres_nn hab'_nn hab_nn
  -- rewrite |result - (a'+b')| = |err| via left cancellation
  -- (a'+b') + err - (a'+b') = err  via float32_add_sub_cancel_left
  have hcancel := float32_add_sub_cancel_left (a' + b') err hab'_nn herr_nn
  rw [hcancel] at htri
  -- step 3: |(a'+b') - (a+b)| ≤ |a'-a| + |b'-b|
  have hdiff : ((a' + b') - (a + b)).abs ≤ (a' - a).abs + (b' - b).abs := by
    rw [float32_add_sub_decomp _ _ _ _ ha_nn hb_nn ha'_nn hb'_nn]
    exact float32_abs_triangle _ _ ha'a_nn hb'b_nn
  -- |result-(a+b)| ≤ err.abs + (|a'-a| + |b'-b|)
  have hdiff_abs := float32_abs_notNaN _ (float32_sub_notNaN _ _ hab'_nn hab_nn)
  have hstep4 : (a' + b' + err - (a + b)).abs ≤ err.abs + ((a' - a).abs + (b' - b).abs) :=
    float32_le_trans _ _ _
      (float32_abs_notNaN _ (float32_sub_notNaN _ _ hres_nn hab_nn))
      (float32_add_notNaN _ _ herr_abs_nn hdiff_abs)
      (float32_add_notNaN _ _ herr_abs_nn (float32_add_notNaN _ _ ha'a_abs hb'b_abs))
      htri
      (float32_add_le_add _ _ _ _ herr_abs_nn herr_abs_nn hdiff_abs
        (float32_add_notNaN _ _ ha'a_abs hb'b_abs)
        (float32_le_refl_notNaN _ herr_abs_nn) hdiff)
  -- bound each term individually
  have hbound : err.abs + ((a' - a).abs + (b' - b).abs) ≤
      0.5 * (machineEpsilon dtype).getD 0 * (a' + b').abs +
      (0.5 * (machineEpsilon dtype).getD 0 * a.abs +
       0.5 * (machineEpsilon dtype).getD 0 * b.abs) :=
    float32_add_le_add _ _ _ _
      herr_abs_nn (float32_mul_notNaN _ _ hu_nn hab'_abs)
      (float32_add_notNaN _ _ ha'a_abs hb'b_abs)
      (float32_add_notNaN _ _ (float32_mul_notNaN _ _ hu_nn ha_abs)
                              (float32_mul_notNaN _ _ hu_nn hb_abs))
      herr
      (float32_add_le_add _ _ _ _ ha'a_abs (float32_mul_notNaN _ _ hu_nn ha_abs)
        hb'b_abs (float32_mul_notNaN _ _ hu_nn hb_abs) ha' hb')
  -- rearrange using distributivity and commutativity
  have hsum_nn : (a.abs + b.abs + (a' + b').abs).isNaN = false :=
    float32_add_notNaN _ _ (float32_add_notNaN _ _ ha_abs hb_abs) hab'_abs
  have hmul_nn := float32_mul_notNaN _ _ hu_nn hsum_nn
  have hrearr : 0.5 * (machineEpsilon dtype).getD 0 * (a' + b').abs +
      (0.5 * (machineEpsilon dtype).getD 0 * a.abs +
       0.5 * (machineEpsilon dtype).getD 0 * b.abs) =
      0.5 * (machineEpsilon dtype).getD 0 * (a.abs + b.abs + (a' + b').abs) := by
    rw [float32_mul_add3 _ _ _ _ hu_nn ha_abs hb_abs hab'_abs]
    rw [float32_add_comm (0.5 * (machineEpsilon dtype).getD 0 * (a' + b').abs)
        (0.5 * (machineEpsilon dtype).getD 0 * a.abs +
         0.5 * (machineEpsilon dtype).getD 0 * b.abs)
        (float32_mul_notNaN _ _ hu_nn hab'_abs)
        (float32_add_notNaN _ _ (float32_mul_notNaN _ _ hu_nn ha_abs)
                                (float32_mul_notNaN _ _ hu_nn hb_abs))]
  -- chain all steps
  rw [hrearr] at hbound
  exact float32_le_trans _ _ _
    (float32_abs_notNaN _ (float32_sub_notNaN _ _ hres_nn hab_nn))
    (float32_add_notNaN _ _ herr_abs_nn (float32_add_notNaN _ _ ha'a_abs hb'b_abs))
    hmul_nn
    hstep4 hbound
