"""btc_dashboard.data — data layer: API clients per venue.

Connectors
----------
- spot       (Binance, Coinbase)
- futures    (Binance, Deribit)
- options    (Deribit, Binance)
- liquidity  (Binance, Deribit, Coinbase)
- flows      (Binance, Deribit, Coinbase)
- coinglass  (cross-exchange aggregations — requires COINGLASS_API_KEY)

Each module returns DataFrames conformed to a per-dataset schema. No
cross-venue merging happens at this layer.
"""
