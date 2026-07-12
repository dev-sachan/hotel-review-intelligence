# Limitations & future work

Honest scope boundaries, and what we would build next.

## Limitations

1. **Synthetic, template-based corpus.** The 87-sentence structure makes the NLP layer
   trivially auditable but also means the language is less varied than real reviews. The
   architecture generalizes to free text (it would classify more unique sentences), but the
   *demonstrated* accuracy is on templated language.

2. **One sentiment value per sentence.** A sentence carrying two aspects gets one polarity
   for both. Real reviews contain intra-sentence contradictions ("great location but filthy
   room") that a clause-level or aspect-conditioned sentiment model would split better. Our
   contradiction handling operates at the hotel×aspect level (across reviews), not within a
   single sentence.

3. **Budget is inferred from star category, not price.** Without nightly rates, the
   budget↔hotel match is a soft proxy. Real rate data would make the value dimension and
   budget fit far sharper.

4. **Calendar seasons ignore hemisphere and local climate.** A Sydney hotel's "summer" is
   December. We detect calendar-periodic drift, which is the right frame for a booking
   calendar but not for local-weather reasoning.

5. **Keyword-rule profile parsing is high-precision but bounded.** It reads the profile
   vocabulary in this dataset very well; a profile phrased in unusual language would lean on
   the embedding backstop, which we measured to be noisier on this taxonomy.

6. **The optional LLM polish is cosmetic.** It only rephrases justifications and is
   validated against fabrication. It adds no reasoning to the ranking (by design), and on a
   pure-CPU machine it noticeably slows the batch run.

7. **No online learning / feedback loop.** Recommendations are computed from the static
   corpus; there is no click-through or booking feedback adjusting the weights.

## Future work

- **Aspect-conditioned sentiment** (ABSA) to resolve intra-sentence contradictions.
- **Rate integration** for a true value model and hard budget filtering.
- **Learned dimension weights** — replace hand-tuned composite weights with weights fit to
  held-out booking/click outcomes.
- **Hemisphere-aware seasonality** using hotel geolocation.
- **Confidence intervals on scores** surfaced in the UI (bootstrap over the weighted
  reviews), so a "3.9 ± 0.6" reads differently from a "3.9 ± 0.1".
- **A/B-testable explanation styles** — template vs LLM vs hybrid — measured on user trust.
- **Incremental ingestion** so new reviews update only the affected hotel×aspect cells.
