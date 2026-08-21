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


import TensorLib.Dtype
import TensorLib.Float
import Mathlib.Order.Basic
import Mathlib.Tactic.CasesM
import Std.Tactic.BVDecide

-- Silence stylistic proof-hygiene linters (unused simp lemmas / redundant tactic
-- branches inside `first`/`try` combinators / unreferenced binders). These are
-- cosmetic; the genuine `declaration uses sorry` notices are intentionally kept.
set_option linter.unusedSimpArgs false
set_option linter.unusedTactic false
set_option linter.unreachableTactic false
set_option linter.unusedVariables false

namespace TensorLib

-- Floats can be a partial order because they satisfy
-- reflexivity, antisymmetric, and transitivity.
-- decode converts a ByteArray to the Float32 value it represents for this dtype.
class POrd (dtype : Dtype) where
  decode  : ByteArray → Float32
  le      : ByteArray → ByteArray → Bool
  notNaN  : ByteArray → Bool
  reflexivity  : ∀ x, notNaN x = true → le x x = true
  -- The `¬(both zero)` guard is required for soundness: IEEE gives +0.0 ≤ -0.0 and
  -- -0.0 ≤ +0.0, yet `decode` of the two bit patterns are distinct Float32 values, so
  -- without this guard the law would be false. See `Float32.le_antisymm_of_not_nan`.
  antisymm     : ∀ x y, notNaN x = true → notNaN y = true → le x y = true → le y x = true →
                 ¬(decode x == 0 ∧ decode y == 0) → decode x = decode y
  transitivity : ∀ x y z, notNaN x = true → notNaN y = true → notNaN z = true → le x y = true → le y z = true → le x z = true


-- For any non-NaN Float32 value f, f <= f evaluates to true.
-- We condition on non-NaN because NaN <= NaN is false in IEEE 754.
-- Proof strategy: unfold Float32.le through the model chain
-- (Float32.le → Float32.Model.le → UnpackedFloat.le → compare),
-- then case split on the four UnpackedFloat constructors.
-- The notANumber case is dismissed by the non-NaN hypothesis.
-- The remaining cases (infinity, zero, finite) each reduce to
-- compare s s = eq which holds since compare is reflexive on Sign and Nat/Int.
set_option linter.unusedSimpArgs false in
theorem Float32.le_refl_of_not_nan (f : Float32) (h : f.isNaN = false) : f <= f := by
  -- Float32.le is Bool-valued; LE Float32 wraps it as (f.le g = true)
  show Float32.le f f = true
  -- Float32.le f f = decide (f.toModel <= f.toModel)
  unfold Float32.le
  -- strip the decide wrapper: decide p = true iff p
  rw [decide_eq_true_eq]
  -- Float32.Model.le is Bool-valued; LE Float32.Model wraps it as (a.le b = true)
  show Float32.Model.le f.toModel f.toModel = true
  -- Float32.Model.le a b = a.unpack.le b.unpack
  unfold Float32.Model.le
  -- UnpackedFloat.le a b = Option.any isLE (a.compare b)
  unfold Float.Model.UnpackedFloat.le
  -- case split on the four constructors of UnpackedFloat
  -- use h_eq to get an explicit equality in the notANumber case
  cases h_eq : f.toModel.unpack with
  | infinity s1 =>
    -- compare (infinity s1) (infinity s1) = compare s1 s1; case split on Sign
    -- Sign has two concrete constructors so decide closes each case
    simp [Float.Model.UnpackedFloat.compare]
    cases s1 <;> decide
  | notANumber =>
    -- h_eq : f.toModel.unpack = notANumber, so f.isNaN = true, contradicts h
    simp [Float32.isNaN, Float32.Model.isNaN, Float.Model.UnpackedFloat.isNaN, h_eq] at h
  | zero s2 =>
    -- compare (zero s2) (zero s2) = eq, closes directly
    simp [Float.Model.UnpackedFloat.compare]
  | finite s3 m e hm =>
    -- case split on Sign; for each sign, compare e e = eq and compare m m = eq
    -- because compare a a = eq follows from lt_irrefl (a < a is false)
    simp [Float.Model.UnpackedFloat.compare]
    have hm_eq : compare m m = .eq := by simp [lt_irrefl]
    have he_eq : compare e e = .eq := by simp [lt_irrefl]
    cases s3 <;> simp [Option.any, Ordering.isLE, Ordering.then, he_eq, hm_eq]


-- The sign bit round-trips: ofBitVec then toBitVec is the identity on a 1-bit vector.
private lemma sign_ofBitVec_toBitVec (b : BitVec 1) :
    (Float.Model.UnpackedFloat.Sign.ofBitVec b).toBitVec = b := by
  simp only [Float.Model.UnpackedFloat.Sign.ofBitVec]
  split <;> rename_i hb <;>
    simp only [Float.Model.UnpackedFloat.Sign.toBitVec] <;> bv_decide

-- pack ∘ unpack = id on valid bit patterns (the model's canonicalization round-trip,
-- which the core library does not provide). This is unpack injectivity: it lets us
-- conclude bit-equality from equal unpacked values, and is the one lemma standing
-- between us and a fully-proved `POrd` antisymmetry (see `le_antisymm_of_not_nan`).
--
-- STATUS: 4 of the 5 constructor cases are proved below — infinity, NaN (via the
-- format's canonical-NaN validity hypothesis), zero, and subnormal finite. Only the
-- normal-finite case remains (`sorry`); it needs the biased-exponent reconstruction
-- `(↑ev.toNat - bias) + bias = ev.toNat` together with `(1 ++ mantissa).log2 = 23`.
private lemma pack_unpack_bv (bv : BitVec 32) (h : Float.Model.Format.binary32.Valid bv) :
    Float.Model.UnpackedFloat.pack Float.Model.Format.binary32
      (Float.Model.UnpackedFloat.unpack Float.Model.Format.binary32 bv) = bv := by
  unfold Float.Model.UnpackedFloat.unpack
  simp only [Float.Model.Format.binary32]
  split_ifs with h1 h2 h3 h4
  · -- exponent all-ones, mantissa 0 → +/- infinity
    simp only [Float.Model.UnpackedFloat.pack, Float.Model.UnpackedFloat.packedInfinity,
               Float.Model.UnpackedFloat.packComponents, sign_ofBitVec_toBitVec,
               Float.Model.UnpackedFloat.unpackSign, Float.Model.UnpackedFloat.unpackExponent,
               Float.Model.UnpackedFloat.unpackMantissa, Float.Model.Format.binary32] at h1 h2 ⊢
    bv_decide
  · -- exponent all-ones, mantissa ≠ 0 → NaN; validity forces bv = canonical NaN
    exact (h.eq_packedNaN h1 h2).symm
  · -- exponent 0, mantissa 0 → +/- zero
    simp only [Float.Model.UnpackedFloat.pack, Float.Model.UnpackedFloat.packedZero,
               Float.Model.UnpackedFloat.packComponents, sign_ofBitVec_toBitVec,
               Float.Model.UnpackedFloat.unpackSign, Float.Model.UnpackedFloat.unpackExponent,
               Float.Model.UnpackedFloat.unpackMantissa, Float.Model.Format.binary32] at h3 h4 ⊢
    bv_decide
  · -- exponent 0, mantissa ≠ 0 → subnormal finite (model exponent -149)
    have hm : (@Float.Model.UnpackedFloat.unpackMantissa Float.Model.Format.binary32 bv).toNat
                < 2 ^ 23 := (@Float.Model.UnpackedFloat.unpackMantissa Float.Model.Format.binary32 bv).isLt
    have hmpos : 0 < (@Float.Model.UnpackedFloat.unpackMantissa Float.Model.Format.binary32 bv).toNat :=
      Nat.pos_of_ne_zero (fun hz => h4 (BitVec.toNat_inj.mp (by simpa using hz)))
    have hlog : (@Float.Model.UnpackedFloat.unpackMantissa Float.Model.Format.binary32 bv).toNat.log2
                < 23 := (Nat.log2_lt (by omega)).mpr hm
    simp only [h3, Float.Model.UnpackedFloat.pack, Float.Model.Format.binary32,
               Float.Model.Format.exponentBias, Float.Model.Format.mantissaBits]
    split_ifs with h_a h_b
    · exact absurd h_a (by decide)     -- not overflow (biased exponent is 1)
    · exact absurd h_b (by omega)      -- not normal (mantissa.log2 + 1 ≤ 23)
    · simp only [Float.Model.UnpackedFloat.packComponents, sign_ofBitVec_toBitVec,
                 BitVec.ofNat_toNat, BitVec.setWidth_eq,
                 Float.Model.UnpackedFloat.unpackSign, Float.Model.UnpackedFloat.unpackExponent,
                 Float.Model.UnpackedFloat.unpackMantissa, Float.Model.Format.binary32] at h3 ⊢
      bv_decide
  · -- exponent normal (≠ all-ones, ≠ 0) → normal finite (implicit leading bit).
    -- The remaining case: biased-exponent reconstruction + `(1 ++ mv).log2 = 23`.
    -- The other four constructors (infinity, NaN, zero, subnormal) are proved above.
    sorry

-- Float32.le is antisymmetric for non-NaN, non-both-zero values.
-- The +0/-0 edge case: IEEE 754 defines +0.0 <= -0.0 and -0.0 <= +0.0,
-- but they have different bit patterns so +0.0 ≠ -0.0 as Float32 values.
-- We exclude this case with hNotBothZero.
-- Proof strategy (once `pack_unpack_bv` is fully closed): `compare_swap` forces
-- `≤` in both directions to `compare = some .eq`, which (with hNotBothZero ruling
-- out the ±0 case) yields `a.toModel.unpack = b.toModel.unpack`; then `pack_unpack_bv`
-- (unpack injectivity) lifts that to `a.toModel = b.toModel`, hence `a = b`.
-- BLOCKED ON: the normal-finite case of `pack_unpack_bv` above (the only remaining gap).
theorem Float32.le_antisymm_of_not_nan (a b : Float32)
  (ha : a.isNaN = false) (hb : b.isNaN = false)
  (h1 : a <= b) (h2 : b <= a)
  (hNotBothZero : ¬(a == 0 ∧ b == 0)) : a = b := by
  sorry

-- Helper: (compare m1 m2).isLE = true ↔ m1 ≤ m2 for Nat.
-- Uses Lean core Nat.compare_eq_lt/eq/gt which match the instOrdNat used by cases.
private lemma nat_isLE_iff_le {m1 m2 : Nat} :
    (compare m1 m2).isLE = true ↔ m1 ≤ m2 := by
  constructor
  · intro h
    cases hc : compare m1 m2 with
    | gt => simp [Ordering.isLE, hc] at h
    | lt => have := Nat.compare_eq_lt.mp hc; omega
    | eq => have := Nat.compare_eq_eq.mp hc; omega
  · intro h
    cases hc : compare m1 m2 with
    | lt | eq => simp [Ordering.isLE, hc]
    | gt =>
      -- compare = .gt means m2 < m1, contradicts h : m1 ≤ m2
      have := Nat.compare_eq_gt.mp hc; omega

-- Helper: lexicographic transitivity for (Int exponent, Nat mantissa) pairs.
-- Uses Lean core Int.compare_eq_lt/eq/gt (instOrdInt) to avoid the Ord instance
-- diamond between instOrdInt (Lean core) and LinearOrder.toOrd (Mathlib).
set_option linter.unusedSimpArgs false in
private lemma finite_lex_trans (e1 e2 e3 : Int) (m1 m2 m3 : Nat)
    (h1 : ((compare e1 e2).then (compare m1 m2)).isLE = true)
    (h2 : ((compare e2 e3).then (compare m2 m3)).isLE = true) :
    ((compare e1 e3).then (compare m1 m3)).isLE = true := by
  cases h12 : compare e1 e2 <;> cases h23 : compare e2 e3 <;>
  simp only [h12, h23, Ordering.then, Ordering.isLE] at h1 h2 ⊢
  -- e1 < e2, e2 < e3 → e1 < e3
  · have he12 := Int.compare_eq_lt.mp h12
    have he23 := Int.compare_eq_lt.mp h23
    simp [Int.compare_eq_lt.mpr (Int.lt_trans he12 he23)]
  -- e1 < e2, e2 = e3 → e1 < e3
  · have he12 := Int.compare_eq_lt.mp h12
    have he23 := Int.compare_eq_eq.mp h23
    simp [Int.compare_eq_lt.mpr (he23 ▸ he12)]
  -- e1 < e2, e2 > e3 → h2 false
  · simp [Ordering.isLE] at h2
  -- e1 = e2, e2 < e3 → e1 < e3
  · have he12 := Int.compare_eq_eq.mp h12
    have he23 := Int.compare_eq_lt.mp h23
    simp [Int.compare_eq_lt.mpr (he12 ▸ he23)]
  -- e1 = e2, e2 = e3 → e1 = e3 and m1 ≤ m3
  · have he12 := Int.compare_eq_eq.mp h12
    have he23 := Int.compare_eq_eq.mp h23
    have lm12 : m1 ≤ m2 := nat_isLE_iff_le.mp h1
    have lm23 : m2 ≤ m3 := nat_isLE_iff_le.mp h2
    rw [Int.compare_eq_eq.mpr (he12.trans he23)]
    simp [Ordering.then]
    exact nat_isLE_iff_le.mpr (by omega)
  -- e1 = e2, e2 > e3 → h2 false
  · simp [Ordering.isLE] at h2
  -- e1 > e2 → h1 false in all sub-cases
  · simp [Ordering.isLE] at h1
  · simp [Ordering.isLE] at h1
  · simp [Ordering.isLE] at h1

-- Same as finite_lex_trans but for negative floats.
-- Negative numbers: bigger magnitude = more negative = smaller value, so comparison is swapped.
-- If neg1 ≤ neg2 ≤ neg3 then (e1,m1) ≥ (e2,m2) ≥ (e3,m3) in the magnitude ordering.
set_option linter.unusedSimpArgs false in
private lemma finite_lex_trans_neg (e1 e2 e3 : Int) (m1 m2 m3 : Nat)
    (h1 : ((compare e1 e2).then (compare m1 m2)).swap.isLE = true)
    (h2 : ((compare e2 e3).then (compare m2 m3)).swap.isLE = true) :
    ((compare e1 e3).then (compare m1 m3)).swap.isLE = true := by
  cases h12 : compare e1 e2 <;> cases h23 : compare e2 e3 <;>
  simp only [h12, h23, Ordering.then, Ordering.isLE, Ordering.swap] at h1 h2 ⊢
  -- e1 < e2: .lt.swap.isLE = .gt.isLE = false, h1 is impossible
  · simp at h1
  · simp at h1
  · simp at h1
  -- e1 = e2, e2 < e3: h2 is false
  · simp at h2
  -- e1 = e2, e2 = e3: e1 = e3, m3 ≤ m2 ≤ m1 so m3 ≤ m1
  · have he : e1 = e3 := (Int.compare_eq_eq.mp h12).trans (Int.compare_eq_eq.mp h23)
    -- h1 is (compare m1 m2).swap.isLE = true (expanded), meaning m2 ≤ m1
    have lm21 : m2 ≤ m1 := by
      cases hc : compare m1 m2 with
      | gt => have := Nat.compare_eq_gt.mp hc; omega
      | eq => have := Nat.compare_eq_eq.mp hc; omega
      | lt => simp [hc] at h1  -- .lt.swap.isLE = false, h1 is impossible
    -- h2 is (compare m2 m3).swap.isLE = true, meaning m3 ≤ m2
    have lm32 : m3 ≤ m2 := by
      cases hc : compare m2 m3 with
      | gt => have := Nat.compare_eq_gt.mp hc; omega
      | eq => have := Nat.compare_eq_eq.mp hc; omega
      | lt => simp [hc] at h2
    rw [Int.compare_eq_eq.mpr he]
    simp only [Ordering.then, Ordering.swap, Ordering.isLE]
    -- goal: (compare m1 m3).swap.isLE = true, i.e., m3 ≤ m1
    cases hc13 : compare m1 m3 with
    | gt => simp [hc13]  -- m3 < m1, .gt.swap = .lt, isLE = true
    | eq => simp [hc13]  -- m3 = m1, .eq.swap = .eq, isLE = true
    | lt =>
      have := Nat.compare_eq_lt.mp hc13  -- m1 < m3, contradicts m3 ≤ m1
      omega
  -- e1 = e2, e2 > e3: e1 > e3
  · have he12 := Int.compare_eq_eq.mp h12
    have he23 := Int.compare_eq_gt.mp h23
    simp [Int.compare_eq_gt.mpr (he12 ▸ he23)]
  -- e1 > e2, e2 < e3: h2 is false
  · simp at h2
  -- e1 > e2, e2 = e3: e1 > e3
  · have he23 := Int.compare_eq_eq.mp h23
    have he12 := Int.compare_eq_gt.mp h12
    simp [Int.compare_eq_gt.mpr (he23 ▸ he12)]
  -- e1 > e2, e2 > e3: e3 < e2 < e1 so e3 < e1
  · have he12 := Int.compare_eq_gt.mp h12
    have he23 := Int.compare_eq_gt.mp h23
    simp [Int.compare_eq_gt.mpr (Int.lt_trans he23 he12)]

-- Transitivity of Float32.le for non-NaN values: a ≤ b → b ≤ c → a ≤ c.
-- Proof: unfold to UnpackedFloat.compare, case split all three constructors,
-- use finite_lex_trans/finite_lex_trans_neg for the finite cases.
set_option linter.unusedSimpArgs false in
set_option maxHeartbeats 800000 in
theorem Float32.le_trans_of_not_nan (a b c : Float32)
    (ha : a.isNaN = false) (hb : b.isNaN = false) (hc : c.isNaN = false)
    (h1 : a <= b) (h2 : b <= c) : a <= c := by
  -- Reduce `a ≤ b`, `b ≤ c` and the goal to `UnpackedFloat.compare … |>.any isLE = true`.
  show Float32.le a c = true
  have h1' : Float32.le a b = true := h1
  have h2' : Float32.le b c = true := h2
  unfold Float32.le at *
  simp only [decide_eq_true_eq] at *
  show Float32.Model.le a.toModel c.toModel = true
  have sh1 : Float32.Model.le a.toModel b.toModel = true := h1'
  have sh2 : Float32.Model.le b.toModel c.toModel = true := h2'
  unfold Float32.Model.le at *
  unfold Float.Model.UnpackedFloat.le at *
  -- Expose the isNaN hypotheses in terms of the unpacked values so the constructor
  -- case split can discharge the notANumber cases.
  simp only [Float32.isNaN, Float32.Model.isNaN, Float32.Model.unpack] at ha hb hc
  cases h_a : a.toModel.unpack <;>
  cases h_b : b.toModel.unpack <;>
  cases h_c : c.toModel.unpack <;>
  -- notANumber cases contradict ha/hb/hc; every other case survives.
  simp_all only [Float.Model.UnpackedFloat.isNaN] <;>
  -- Split every remaining sign so all comparisons become concrete (except finite exponents).
  casesm* Float.Model.UnpackedFloat.Sign <;>
  -- Evaluate the (now sign-concrete) comparisons in the hypotheses and goal.
  (try simp only [Float.Model.UnpackedFloat.compare, Option.any, Ordering.isLE,
                  Ordering.then, Ordering.swap] at sh1 sh2 ⊢) <;>
  first
  -- finite/finite/finite with matching signs: lexicographic transitivity on (exponent, mantissa).
  | exact finite_lex_trans _ _ _ _ _ _ sh1 sh2
  | exact finite_lex_trans_neg _ _ _ _ _ _ sh1 sh2
  -- everything else: either directly true, or the hypotheses are contradictory.
  | assumption
  | decide
  | simp_all
  | omega

-- `compare` is antisymmetric: swapping the arguments swaps the ordering.
-- Holds for all values (including NaN, where both sides are `none`).
-- Proof: split constructors and signs; the finite cases use `Int/Nat.compare_swap`
-- together with `Ordering.swap_then`, and the rest evaluate concretely.
set_option linter.unusedSimpArgs false in
private lemma compare_swap (u v : Float.Model.UnpackedFloat) :
    u.compare v = (v.compare u).map Ordering.swap := by
  cases u <;> cases v <;>
  (first | casesm* Float.Model.UnpackedFloat.Sign | skip)
  all_goals (
    try dsimp only [Float.Model.UnpackedFloat.compare]
    -- finite/finite same-sign: rewrite via lexicographic swap; other cases are concrete.
    try simp only [Option.map_some, Option.map_none,
                   Ordering.swap_then, Ordering.swap_swap, Int.compare_swap, Nat.compare_swap]
    try rfl
    try decide
    try simp_all)

-- `≤` and the strict `<` in the reverse direction cannot both hold.
-- No non-NaN hypotheses are needed: if either value is NaN then `a ≤ b` is already false
-- (NaN compares to `none`, and `none.any isLE = false`), so the hypotheses are contradictory.
-- Proof: `a ≤ b` gives `(compare a b).any isLE`, and `b < a` gives `compare b a = some .lt`.
-- By `compare_swap`, `compare a b = (some .lt).swap = some .gt`, whose `isLE` is `false`.
theorem Float32.le_lt_false (a b : Float32) (h1 : a ≤ b) (h2 : b < a) : False := by
  -- a ≤ b  ⇒  (a.unpack.compare b.unpack).any isLE = true
  have h1' : Float32.le a b = true := h1
  unfold Float32.le at h1'
  simp only [decide_eq_true_eq] at h1'
  have hle : Float32.Model.le a.toModel b.toModel = true := h1'
  unfold Float32.Model.le Float.Model.UnpackedFloat.le at hle
  -- b < a  ⇒  (b.unpack.compare a.unpack) = some .lt
  have h2' : Float32.lt b a = true := h2
  unfold Float32.lt at h2'
  simp only [decide_eq_true_eq] at h2'
  have hlt : Float32.Model.lt b.toModel a.toModel = true := h2'
  unfold Float32.Model.lt Float.Model.UnpackedFloat.lt at hlt
  rw [beq_iff_eq] at hlt
  -- rewrite compare a b via the swap of compare b a = some .lt
  rw [compare_swap a.toModel.unpack b.toModel.unpack, hlt] at hle
  simp only [Option.map_some, Ordering.swap, Option.any, Ordering.isLE] at hle
  exact absurd hle (by decide)

instance : POrd .float32 where
  decode  := Float32.ofLEByteArray!
  le      := fun x y => decide (Float32.ofLEByteArray! x ≤ Float32.ofLEByteArray! y)
  notNaN  := fun x => !(Float32.ofLEByteArray! x).isNaN
  reflexivity := by
    intro x hx
    simp only [decide_eq_true_eq]
    simp at hx
    exact Float32.le_refl_of_not_nan _ hx
  antisymm := by
    intro x y hx hy hxy hyx
    -- TODO: provide hNotBothZero to le_antisymm_of_not_nan or handle ±0 case
    sorry
  transitivity := by
    intro x y z hx hy hz hxy hyz
    simp at hxy hyz hx hy hz
    simp only [decide_eq_true_eq]
    exact Float32.le_trans_of_not_nan _ _ _ hx hy hz hxy hyz

instance : POrd .bfloat16 where
  decode  := fun x => (Dtype.byteArrayToBFloat16 .bfloat16 x).toOption.getD Float32.quietNaN
  le      := fun x y =>
    let fx := (Dtype.byteArrayToBFloat16 .bfloat16 x).toOption.getD Float32.quietNaN
    let fy := (Dtype.byteArrayToBFloat16 .bfloat16 y).toOption.getD Float32.quietNaN
    decide (fx ≤ fy)
  notNaN  := fun x => !((Dtype.byteArrayToBFloat16 .bfloat16 x).toOption.getD Float32.quietNaN).isNaN
  reflexivity := by
    intro x hx
    simp only [decide_eq_true_eq]
    simp at hx
    exact Float32.le_refl_of_not_nan _ hx
  antisymm := by
    intro x y hx hy hxy hyx
    -- TODO: provide hNotBothZero to le_antisymm_of_not_nan or handle ±0 case
    sorry
  transitivity := by
    intro x y z hx hy hz hxy hyz
    simp at hxy hyz hx hy hz
    simp only [decide_eq_true_eq]
    exact Float32.le_trans_of_not_nan _ _ _ hx hy hz hxy hyz

instance : POrd .float16 where
  decode  := fun x => (Dtype.byteArrayToFloat16 .float16 x).toOption.getD Float32.quietNaN
  le      := fun x y =>
    let fx := (Dtype.byteArrayToFloat16 .float16 x).toOption.getD Float32.quietNaN
    let fy := (Dtype.byteArrayToFloat16 .float16 y).toOption.getD Float32.quietNaN
    decide (fx ≤ fy)
  notNaN  := fun x => !((Dtype.byteArrayToFloat16 .float16 x).toOption.getD Float32.quietNaN).isNaN
  reflexivity := by
    intro x hx
    simp only [decide_eq_true_eq]
    simp at hx
    exact Float32.le_refl_of_not_nan _ hx
  antisymm := by
    intro x y hx hy hxy hyx
    -- TODO: provide hNotBothZero to le_antisymm_of_not_nan or handle +-0 case
    sorry
  transitivity := by
    intro x y z hx hy hz hxy hyz
    simp at hxy hyz hx hy hz
    simp only [decide_eq_true_eq]
    exact Float32.le_trans_of_not_nan _ _ _ hx hy hz hxy hyz

instance : POrd .float8_e4m3 where
  decode  := fun x => (Dtype.decodeFloat8E4M3 x).toOption.getD Float32.quietNaN
  le      := fun x y =>
    let fx := (Dtype.decodeFloat8E4M3 x).toOption.getD Float32.quietNaN
    let fy := (Dtype.decodeFloat8E4M3 y).toOption.getD Float32.quietNaN
    decide (fx ≤ fy)
  notNaN  := fun x => !((Dtype.decodeFloat8E4M3 x).toOption.getD Float32.quietNaN).isNaN
  reflexivity := by
    intro x hx
    simp only [decide_eq_true_eq]
    simp at hx
    exact Float32.le_refl_of_not_nan _ hx
  antisymm := by
    intro x y hx hy hxy hyx
    -- TODO: provide hNotBothZero to le_antisymm_of_not_nan or handle ±0 case
    sorry
  transitivity := by
    intro x y z hx hy hz hxy hyz
    simp at hxy hyz hx hy hz
    simp only [decide_eq_true_eq]
    exact Float32.le_trans_of_not_nan _ _ _ hx hy hz hxy hyz

instance : POrd .float8_e5m2 where
  decode  := fun x => (Dtype.decodeFloat8E5M2 x).toOption.getD Float32.quietNaN
  le      := fun x y =>
    let fx := (Dtype.decodeFloat8E5M2 x).toOption.getD Float32.quietNaN
    let fy := (Dtype.decodeFloat8E5M2 y).toOption.getD Float32.quietNaN
    decide (fx ≤ fy)
  notNaN  := fun x => !((Dtype.decodeFloat8E5M2 x).toOption.getD Float32.quietNaN).isNaN
  reflexivity := by
    intro x hx
    simp only [decide_eq_true_eq]
    simp at hx
    exact Float32.le_refl_of_not_nan _ hx
  antisymm := by
    intro x y hx hy hxy hyx
    -- TODO: provide hNotBothZero to le_antisymm_of_not_nan or handle ±0 case
    sorry
  transitivity := by
    intro x y z hx hy hz hxy hyz
    simp at hxy hyz hx hy hz
    simp only [decide_eq_true_eq]
    exact Float32.le_trans_of_not_nan _ _ _ hx hy hz hxy hyz

instance : POrd .float8_e3m4 where
  decode  := fun x => (Dtype.decodeFloat8E3M4 x).toOption.getD Float32.quietNaN
  le      := fun x y =>
    let fx := (Dtype.decodeFloat8E3M4 x).toOption.getD Float32.quietNaN
    let fy := (Dtype.decodeFloat8E3M4 y).toOption.getD Float32.quietNaN
    decide (fx ≤ fy)
  notNaN  := fun x => !((Dtype.decodeFloat8E3M4 x).toOption.getD Float32.quietNaN).isNaN
  reflexivity := by
    intro x hx
    simp only [decide_eq_true_eq]
    simp at hx
    exact Float32.le_refl_of_not_nan _ hx
  antisymm := by
    intro x y hx hy hxy hyx
    -- TODO: provide hNotBothZero to le_antisymm_of_not_nan or handle ±0 case
    sorry
  transitivity := by
    intro x y z hx hy hz hxy hyz
    simp at hxy hyz hx hy hz
    simp only [decide_eq_true_eq]
    exact Float32.le_trans_of_not_nan _ _ _ hx hy hz hxy hyz


end TensorLib
