import SciLean

namespace TensorLib

def t₁ : Float^[3] := ⊞[1.0,2.0,3.0]

def x₁ := t₁[1]

#eval t₁[-1] -- 3.000000
#eval t₁[4] -- 2.000000

def t₂ : Float^[3]^[2] := ⊞[t₁,t₁]

def t₃ : Float^[3]^[2]^[2] := ⊞[t₂,t₂]

#eval t₃

def t₄ := t₃ + t₃

#eval t₄

def t₅ := t₃ + 1

#eval t₅

def t₆ : Float^[6] := ⊞[1.0,2.0,3.0,1.0,2.0,3.0]

def t₇ : Float^[2, 3] := t₆.reshape (Fin 2 × Fin 3) (by decide)

#eval t₂[1] -- ⊞[1.000000, 2.000000, 3.000000]
#eval t₇[0] -- 1.000000 ?????
#eval t₇[(0,1)]
#eval t₇[0,1]

example :
  ∀ n : ℕ, ∀ t₁ t₂ : Float^[n], ∀ i : Fin n,
  (t₁ + t₂)[i] = t₁[i] + t₂[i] := by
  simp only [SciLean.ArrayType.add_get, implies_true]

example :
  ∀ n : ℕ, ∀ t : Float^[n], ∀ i : Fin n,
  (t.mapMono fun x : Float => x + 1.0)[i] = t[i] + 1.0 := by
  simp only [SciLean.ArrayType.get_mapMono, implies_true]


end TensorLib
