# Solution Summary

**Project:** Hotel Review Intelligence Engine
**Problem statement:** Hotel Review Intelligence Engine (Expedia Group campus hackathon)

## I. The selected problem statement

Build an AI-powered hotel intelligence system that analyzes hotel reviews and ratings to
assess performance over time, detect seasonal and aspect-level sentiment shifts, and
generate personalized, evidence-based hotel recommendations based on user profile, review
context, and semantic intent.

## II. The user / business problem

Travelers face thousands of undifferentiated reviews and star ratings that don't answer the
question they actually have: *"Is this the right hotel **for me**, and can I trust it right
now?"* A solo female traveler, a corporate road-warrior, and a family with toddlers reading
the same 4-star hotel need completely different signals — and a review from two years ago,
or from an unverified account, should not count the same as a recent verified one. Generic
average ratings collapse all of this into one number.

For the business (a booking platform), the same data is an untapped asset: which hotels are
*improving* or *declining*, which have *seasonal* weak spots, and where reviewers
*genuinely disagree* — none of which a mean rating reveals.

## III. The proposed solution

A pipeline that:

1. **Understands the reviews** — decomposes 50,000 reviews into aspect-tagged, sentiment-
   scored sentences across 15 dimensions (cleanliness, service, value, safety, accessibility,
   nightlife, …). A key insight — the corpus reduces to **87 unique template sentences** —
   let us classify once and **audit every label to 100% accuracy**.
2. **Scores hotels defensibly** — recency-weighted (365-day half-life), verified-weighted,
   traveler-type-conditioned, and empirical-Bayes-shrunk hotel×aspect sentiment.
3. **Reasons over time** — Kruskal-Wallis seasonal tests and Spearman trend tests across
   ~1,070 hotel×aspect series, all FDR-corrected and reported with honest confidence tiers.
4. **Handles contradictions** — detects where reviewers split on an aspect and, where
   possible, *explains* the split by reviewer segment (verified vs unverified, traveler
   type).
5. **Personalizes** — parses each traveler profile into weighted priorities and produces a
   **deterministic** top-5 ranking, every recommendation carrying verbatim review evidence
   (real IDs, dates) and honest caveats. AI does perception; arithmetic does ranking, so
   nothing hallucinates.

Delivered as a Streamlit app whose centerpiece compares multiple travelers' recommendations
**side-by-side**, plus a hotel deep-dive, a portfolio-health dashboard, and semantic search.

## IV. Expected value / impact

- **For travelers:** recommendations they can *trust and verify* — matched to their stated
  needs, backed by quotes, and honest about seasonal or contested weaknesses.
- **For the platform:** an early-warning system for declining properties, seasonal-quality
  alerts, and a defensible, reproducible ranking that can be explained to a hotel partner
  down to individual review weights.
- **Technically:** a design that runs end-to-end on a laptop CPU in minutes, with a fully
  auditable NLP layer and a deterministic, schema-validated output contract.
