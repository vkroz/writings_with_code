# CLAUDE.md

This repo contains notebooks with papers and code. 

## Voice and conventions (match these when editing)

- **Compressed and declarative.** Short, high-density sentences. No hedging, no filler transitions ("Moreover," "It is important to note"), no throat-clearing. Cut, don't pad.
- **The "one sentence" motif.** Sections distill to a single line. Keep these; they are the essay's rhythm.
- **Concrete over abstract.** Claims anchor to specific dates, models, and numbers. Keep the specificity.
- **Em-dashes and colons** carry the argument's turns. Intentional style, not a tic to normalize away.

## Hard constraints — easy to violate, do not

- **Preserve every formula exactly.** Math is in LaTeX blocks (`$$ … $$` / `$ … $`): the reward product $r = r_{\text{action}} \times r_{\text{output}}$, the `pass^k` / `pass@k` estimators, the reliability metrics. Do not rewrite, "simplify," reorder, or alter subscripts/symbols — including when touching surrounding prose.
- **Do not silently alter factual claims.** The essay is dense with load-bearing specifics: dates (July 2025 Replit, Feb 2024 Air Canada), figures (GPT-4o 61.2% `pass^1` retail / 35.2% airline; $0.9^8 \approx 43\%$; policy ablation −4.4 retail / −22.4 airline), citations and arXiv links, and the per-year slopes. If you believe one is wrong, **flag it to the user** with your reasoning — never edit it in place.
- **Verify before "correcting" framing.** Forward-dated references and version names (τ²/τ³-bench, model names) are the author's framing, not hallucinations. Check a real source before changing anything.
- **Keep code cells runnable and consistent with their output.** If you edit a code cell, its printed output shown in the notebook must still match. Re-run or update the output; don't leave stale results.

## When the user asks to change the content

Confirm scope first: a copy-edit (tighten prose, fix typos), a structural change (add/cut/reorder a section), and a factual revision are very different asks. Surface and confirm structural and factual changes — don't assume them. Keep the numbered-section structure and the spine intact unless explicitly asked to change them.