# Weight Management Coexistence Review Checklist

Follow this checklist for PRs adding or updating unified weight-stack branches
during legacy/unified coexistence (RFC #9874 section 9.5):

- Identify the coexistence row ID (`C01`-`C19` or `D01`-`D05`, or a new `NEW_ROW`)
  and `legacy_removal` scope in the PR description.
- Keep original legacy statements intact inside the legacy branch.
- Preserve existing test files and `tests/diffusion/weight_management/coexistence/registry.py`;
  add new tests covering unified behavior.
- Model code reads the selector only through
  `weight_management.components.is_unified_stack()` or the structure-construction
  context query, without direct `od_config.diffusion_weight_stack` access.
- Add a site file under `tests/diffusion/weight_management/coexistence/sites/`
  recording every new selector/context read site.
- Do not add entries to the frozen allowlists in `registry.py`.
