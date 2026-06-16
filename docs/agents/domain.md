# Domain Docs

How the engineering skills should consume this repo's domain documentation.

## Before exploring, read these

- `CONTEXT.md` at the repo root.
- `docs/adr/` for ADRs touching the area being changed.

If these files don't exist, proceed silently. The producer skill `/grill-with-docs` creates them lazily when terms or decisions get resolved.

## Layout

This is a single-context repo:

- `/CONTEXT.md`
- `/docs/adr/`

## Use the glossary's vocabulary

When output names a domain concept, use the term as defined in `CONTEXT.md`. Avoid inventing synonyms.

## Flag ADR conflicts

If output contradicts an existing ADR, surface it explicitly rather than silently overriding.
