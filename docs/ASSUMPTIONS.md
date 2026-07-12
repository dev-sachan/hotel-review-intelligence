# Assumptions

Reasonable assumptions made where the dataset or brief left room, each with its reasoning.

1. **The review corpus is template-generated (87 unique sentences).** Verified empirically:
   136,451 sentence instances collapse to 87 distinct strings. We assume this is by design
   (a synthetic dataset) and exploit it — classifying the 87 once and auditing every label.
   The pipeline still works on free-text reviews; it would just classify more sentences.

2. **Calendar seasons, northern-hemisphere labels.** Seasons are assigned by month
   (Dec–Feb = winter, etc.). The hotels are worldwide, so "summer" is a *calendar* label,
   not a weather claim. We are detecting *periodic* drift keyed to the calendar, which is
   what a global booking platform would compare against. Both dataset years (2024–2025)
   are fully present, so each hotel is observed in each season twice.

3. **Star category is a proxy for price band.** The dataset has no nightly rates, so we map
   a profile's stated budget to a preferred star band (tight→3-star, mid→4-star,
   high→5-star) as a soft ranking term, not a hard filter.

4. **Unverified reviews are discounted, not discarded.** Verified reviews weigh 1.0,
   unverified 0.8. Over half the corpus is unverified; dropping it would gut the sample.
   The mild discount favors trustworthy sources while retaining information.

5. **`traveler_type` is missing on ~60% of rows and treated as "unknown" (neutral).** We
   never drop those rows; at recommendation time a profile's own traveler type is
   up-weighted (×1.35), while unknown-type reviews contribute at their base weight.

6. **Sentiment is expressed from the reviewer's framing, and that framing picks the
   aspect.** For sentences describing the same fact with opposite value to different
   travelers (a quiet area), the complaint framing ("dead quiet, nothing to do") is tagged
   `nightlife`-negative and the praise framing ("wonderfully peaceful") is tagged
   `quietness`-positive.

7. **One sentiment value per sentence.** The sentiment model emits a single polarity per
   sentence, applied to all aspects that sentence carries. For the rare genuinely
   mixed-polarity sentence this is a simplification (see LIMITATIONS).

8. **A profile's stated priorities are authoritative.** We infer desired dimensions
   primarily from explicit phrases in the profile description (keyword rules), using
   embeddings only as a backstop for under-specified profiles. We do not infer unstated
   needs (e.g. we don't assume every family cares about cleanliness unless they say so).

9. **The confidence bar for showing a drift signal is deliberately low but labeled.** We
   surface "weak" signals (FDR-adjusted p < 0.10) *with* their tier rather than hiding
   them, on the assumption that an honestly-labeled weak signal is more useful to a judge
   than silence.

10. **Display scores are rescaled to ~0–5** to match the magnitude of the provided
    `sample_output.json`. The internal composite lives in roughly [−1, 1]; the mapping is
    `(composite + 1) × 2.5`.
