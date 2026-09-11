"""
External data providers.

Everything in this package converts a real-world feed into the exact schema
`backend.features` expects, so the models never know or care where a number
came from. That is the whole point of the `features.py` contract: swapping
simulated weather for live weather is a provider change, not a model change.
"""
