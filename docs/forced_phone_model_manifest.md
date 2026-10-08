# Qualified Forced-Phone Model Manifest

The qualified-model acceptance route requires a detached
`forced_phone_model_manifest_v1` file. A model acceptance record points to the
manifest with `resource_manifest_path` and `resource_manifest_sha256`.

The manifest must list `model_weights`, `dictionary`, and `phone_map` in
`required_artifacts`. A tokenizer is also listed when the runtime needs one.
Every artifact entry contains:

- a local `path` and SHA-256 of the exact bytes used;
- an immutable `revision`;
- a license that applies to that artifact kind, explicitly allows commercial
  use, and explicitly has no NonCommercial restriction; and
- a hash-bound `source_path` containing the license/rights evidence.

The `phone_inventory` symbols and canonical hash must match the parsed phone
segments. The `phone_map` entry repeats that inventory hash. Finally,
`training_evidence` is a hash-bound primary record that explicitly states
`trained_or_finetuned_on_cantonese_singing`, names the Cantonese singing
dataset(s), and records its immutable source revision. It also carries a
`source_path` and `source_sha256` for the primary evidence itself; the loader
checks those bytes independently from the training record hash.

The loader resolves relative paths from the manifest directory and verifies
all file bytes before returning normalized provenance. A missing, changed,
mutable, NC-restricted, or unbound resource is a hard failure. Fixture mode
does not bypass this check; the fixture tests create temporary resource files
so the contract is exercised against real bytes.

This schema describes evidence requirements only. It does not make a model
qualified when the upstream primary source fails to establish Cantonese
singing training or redistribution rights.
