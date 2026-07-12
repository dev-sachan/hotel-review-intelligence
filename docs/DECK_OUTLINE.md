# Presentation deck outline (6–8 slides) + demo-video script

Mapped to the judging criteria: problem understanding, AI/data/reasoning, solution design,
innovation, demo evidence, storytelling.

---

### Slide 1 — Title & hook
**Hotel Review Intelligence Engine.** One line: *"50,000 reviews say a hotel is '4 stars.'
That number is useless — because the right hotel for a solo traveler is the wrong hotel for
a family, and last year's review isn't this year's hotel."*

### Slide 2 — The problem (understanding & relevance)
- Travelers drown in undifferentiated reviews; a mean rating hides *who* it's right for and
  *whether it's still true*.
- Business misses signals: which hotels are declining, seasonally weak, or contested.
- The three questions we answer: **Right for me? Trustworthy now? What's the catch?**

### Slide 3 — The insight that shaped everything (innovation)
- 50,000 reviews → **87 unique template sentences**.
- Consequence: classify **once**, **audit every label** (100% accuracy), and move the real
  engineering up to aggregation, temporal stats, and personalization.
- *"We turned an hours-long GPU job into a seconds-long, fully-auditable CPU job."*

### Slide 4 — Architecture (solution design)
- Diagram: ingest → 87-sentence NLP → hotel×aspect scoring → temporal drift +
  contradiction → profile parser → deterministic recommender → validated JSON → app.
- **Design stance:** AI does *perception*; arithmetic does *ranking*. Deterministic,
  reproducible, nothing hallucinates. (Recency ½-life, verified weighting, empirical-Bayes
  shrinkage, Benjamini-Hochberg FDR — one line each.)

### Slide 5 — Reasoning depth (AI/data/reasoning)
- **Temporal:** seasonal (Kruskal-Wallis) + trend (Spearman), FDR-corrected, honest tiers.
- **Contradiction handling:** detect the split, *explain it by reviewer segment*
  ("business travelers rate WiFi +0.55, families −0.20").
- **Personalization:** profile → weighted dims → composite fit, with an evidence floor
  (never recommend on silence).

### Slide 6 — Demo (prototype evidence) → cut to video
- The side-by-side view: P01 solo-culture vs P29 backpacker vs P30 luxury — **0 shared
  hotels**, each pick backed by real quotes.
- Hotel deep-dive: a declining trend annotation; a seasonal heatmap; a contradiction panel.
- Portfolio pulse: improving/declining movers; seasonal alerts.

### Slide 7 — Datasets, inputs & outputs
- Inputs: `hotel_reviews.json` (50k), `user_profiles.json` (50). Optional local LLM
  (Qwen2.5-1.5B) for prose only.
- Output: exact `sample_output.json` schema, validated via Pydantic, for all 50 profiles +
  a richer evidence-carrying format.

### Slide 8 — Value, honesty & what's next
- Value: trustworthy recommendations for travelers; early-warning + partner-explainable
  rankings for the platform.
- Limitations stated plainly (templated corpus, one-sentiment-per-sentence, star-as-price).
- Future: aspect-conditioned sentiment, rate integration, learned weights, hemisphere-aware
  seasons.

---

## Demo video script (3–5 min)

1. **(0:00–0:30) Hook.** Open on the Recommendations tab with four profiles already
   selected. "Same 120 hotels, four different travelers. Watch how the rankings diverge."
2. **(0:30–1:30) Personalization.** Point at P01 (solo, culture) vs P30 (luxury, spa):
   totally different hotels, different star bands. Expand a card — read a real quote with
   its review ID and verified badge. "Every recommendation is evidence, not vibes."
3. **(1:30–2:30) Trust & time.** Switch to Hotel deep-dive. Show a hotel whose rating is
   *declining* (annotation), its seasonal heatmap, and the contradiction panel — "verified
   guests rate this higher than unverified." "We don't average disagreement away; we explain
   it."
4. **(2:30–3:15) Business view.** Portfolio pulse: improving vs declining movers, seasonal
   alerts. "For the platform, this is an early-warning system."
5. **(3:15–4:00) Under the hood.** One line on the 87-sentence insight and the deterministic
   ranking. Show `outputs/submission/P01.json` matching the required schema. "Auditable,
   reproducible, schema-valid."
6. **(4:00–4:30) Close.** "Right for me, trustworthy now, and honest about the catch — from
   50,000 reviews, on a laptop, in minutes."
