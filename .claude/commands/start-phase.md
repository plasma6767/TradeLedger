---
description: Start the next TradeLedger phase (or a specific one), design it conversationally, then build/test/branch/PR it
---

Start work on TradeLedger phase: $ARGUMENTS (if blank, use the next unchecked phase in `docs/PLAN.md`).

1. Read `docs/PLAN.md`, `README.md`, `CLAUDE.md`, and any other files needed
   for context (existing `src/` modules the new phase will build on,
   relevant `tests/` for style conventions).
2. Explore what real data/APIs the phase needs before proposing anything -
   check actual field names, response shapes, and any real data quirks
   (traded players, missing categories, encoding issues) rather than
   assuming.
3. Explain the phase in plain English first - what problem it solves,
   with a concrete worked example using real or realistic numbers -
   before any technical/implementation detail. Then propose a concrete
   module breakdown mirroring the existing `src/` conventions (fetch/parse
   split, pure computation functions, tests alongside).
4. Get explicit buy-in on the approach before writing code. Flag real
   design tradeoffs (data scope, how a metric will actually read to a
   human, computation cost) rather than silently picking one.
5. Once agreed: branch off `main` (`phase<N>/<short-name>`), implement one
   module at a time, testing and committing after each one lands - never
   one giant commit at the end.
6. Validate against real data before considering it done (not just unit
   tests) - run the real pipeline on a real example and sanity-check the
   output against basketball intuition.
7. Update `docs/PLAN.md` to mark the phase complete with a module
   breakdown section matching the style of the existing phases. Check
   whether `README.md`/`CLAUDE.md` need updates too, but don't touch them
   if what's already there still holds.
8. Run the full test suite one last time, then push the branch and open a
   PR with a clear, human-written description (what changed, why, how it
   was validated). Tell the user it's ready for review - don't merge it
   yourself.
