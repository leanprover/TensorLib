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

import Lake
open Lake DSL

package "TensorLib" where
  -- add package configuration options here

lean_lib «TensorLib» where
  -- add library configuration options here

@[default_target]
lean_exe "tensorlib" where
  root := `Main

require aesop from git
  "https://github.com/leanprover-community/aesop" @ "v4.21.0"

require plausible from git
  "https://github.com/leanprover-community/plausible" @ "v4.21.0"

require Cli from git
  "https://github.com/leanprover/lean4-cli.git" @ "v4.21.0"

require importGraph from git
  "https://github.com/leanprover-community/import-graph.git" @ "v4.21.0"
