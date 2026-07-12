"""The aspect taxonomy: 15 dimensions the engine reasons about, plus a `general`
filler bucket.

Each dimension carries three things:
  - `hypothesis`   : the natural-language statement fed to the zero-shot classifier
                     ("This review talks about ...") — this is what the model scores.
  - `descriptors`  : short phrases used to match a *user profile* to this dimension via
                     sentence-embedding cosine similarity (profile understanding).
  - `display`      : human-friendly label for the UI / output.

The vocabulary deliberately mirrors the sample_output.json `desired_dims` style
(`safety`, `local_culture`, `location_central`) so our outputs slot straight into the
expected schema.
"""
from __future__ import annotations

# Ordered so the most "universal" quality aspects come first.
ASPECTS: dict[str, dict] = {
    "cleanliness": {
        "hypothesis": "the cleanliness, hygiene, or housekeeping of the room",
        "descriptors": ["spotlessly clean rooms", "spotless hygiene and housekeeping",
                        "immaculate spacious rooms"],
        "display": "Cleanliness",
    },
    "service": {
        "hypothesis": "the helpfulness, friendliness, or attitude of the staff and service",
        "descriptors": ["attentive personal service", "warm helpful friendly staff",
                        "front desk that anticipates your needs"],
        "display": "Service",
    },
    "luxury": {
        "hypothesis": "luxury, refinement, or a premium five-star experience",
        "descriptors": ["impeccable five-star luxury and refinement",
                        "world-class premium indulgent experience", "privacy and refinement"],
        "display": "Luxury & refinement",
    },
    "value": {
        "hypothesis": "the price, value for money, or affordability",
        "descriptors": ["great value for money on a tight budget",
                        "affordable good bang for the buck", "cheap budget-friendly rates"],
        "display": "Value for money",
    },
    "location_central": {
        "hypothesis": "a central location, walkability, or being close to the main sights",
        "descriptors": ["central walkable base close to everything",
                        "right in the heart of the city near the sights",
                        "central location near the office district"],
        "display": "Central location",
    },
    "local_culture": {
        "hypothesis": "local culture, authentic neighborhoods, markets, or local character",
        "descriptors": ["authentic local culture and markets",
                        "a genuine local neighborhood not a tourist bubble",
                        "local artisans street food and real character"],
        "display": "Local culture",
    },
    "safety": {
        "hypothesis": "safety, security, or how safe the area feels, especially at night",
        "descriptors": ["a safe secure area especially walking back at night",
                        "feeling secure as a solo traveler",
                        "a quiet safe well-lit neighborhood"],
        "display": "Safety",
    },
    "quietness": {
        "hypothesis": "quietness, noise levels, soundproofing, or a peaceful restful room",
        "descriptors": ["a quiet peaceful room with good soundproofing",
                        "a restful room away from street noise",
                        "a quiet room for calls and focus"],
        "display": "Quietness",
    },
    "nightlife": {
        "hypothesis": "nightlife, bars, clubs, or a lively social scene nearby",
        "descriptors": ["lively nightlife bars and clubs nearby",
                        "a buzzing social scene after dark",
                        "great spots for a night out within walking distance"],
        "display": "Nightlife",
    },
    "family_friendly": {
        "hypothesis": "being family-friendly, good for children, kids' facilities, or a pool",
        "descriptors": ["family-friendly with connecting rooms a kids club and a pool",
                        "great facilities for children and toddlers",
                        "spacious family suites and things for kids to do"],
        "display": "Family-friendly",
    },
    "business_facilities": {
        "hypothesis": "business facilities, fast WiFi, a work desk, or meeting rooms",
        "descriptors": ["fast reliable WiFi and a proper work desk",
                        "meeting rooms and business center for work trips",
                        "rock-solid internet for video calls and remote work"],
        "display": "Business & WiFi",
    },
    "wellness_spa": {
        "hypothesis": "a spa, wellness facilities, sauna, massages, or a wellness retreat",
        "descriptors": ["a world-class spa and wellness retreat",
                        "sauna massages and serene relaxation pool",
                        "yoga and wellness treatments to unwind"],
        "display": "Spa & wellness",
    },
    "beach_access": {
        "hypothesis": "the beach, direct beach access, or a beachfront or seaside setting",
        "descriptors": ["direct beach access and a beachfront setting",
                        "waking up to the sea with loungers ready",
                        "a proper beach holiday by the shore"],
        "display": "Beach access",
    },
    "food_dining": {
        "hypothesis": "food, dining, restaurants, breakfast, or the on-site cuisine",
        "descriptors": ["excellent on-site dining and local cuisine",
                        "a delicious extensive breakfast spread",
                        "great food central to the restaurant scene"],
        "display": "Food & dining",
    },
    "accessibility": {
        "hypothesis": "accessibility, step-free access, ramps, lifts, or wheelchair access",
        "descriptors": ["full step-free wheelchair accessibility",
                        "ramps lifts and roll-in showers for reduced mobility",
                        "an accessible room with wide doorways"],
        "display": "Accessibility",
    },
}

GENERAL_LABEL = "general"  # filler / non-aspect sentences (intros, overall verdicts)

ASPECT_KEYS: list[str] = list(ASPECTS.keys())
HYPOTHESES: list[str] = [ASPECTS[a]["hypothesis"] for a in ASPECT_KEYS]
DISPLAY_NAMES: dict[str, str] = {a: ASPECTS[a]["display"] for a in ASPECT_KEYS}


def descriptor_texts() -> list[tuple[str, str]]:
    """(aspect_key, descriptor) pairs for building the profile-matching embedding bank."""
    return [(a, d) for a in ASPECT_KEYS for d in ASPECTS[a]["descriptors"]]
