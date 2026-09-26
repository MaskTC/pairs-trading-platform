"""Transaction cost and slippage model.

Costs are charged in basis points of traded notional, once per position
change (entry, exit, or each leg of a flip). Both commission-like costs
and slippage are modelled as linear in notional, which is the standard
first-order approximation for liquid equities.
"""


class TransactionCostModel:
    """Linear bps cost model.

    Parameters
    ----------
    cost_bps : float
        Explicit costs (commission, fees) per side, in basis points.
    slippage_bps : float
        Assumed adverse price movement per side, in basis points.
    """

    def __init__(self, cost_bps: float = 5.0, slippage_bps: float = 2.0):
        if cost_bps < 0 or slippage_bps < 0:
            raise ValueError("bps values must be non-negative")
        self.cost_bps = float(cost_bps)
        self.slippage_bps = float(slippage_bps)

    @property
    def total_bps(self) -> float:
        """Combined drag per side, in basis points."""
        return self.cost_bps + self.slippage_bps

    def cost(self, notional: float) -> float:
        """Dollar cost of trading ``notional`` dollars (one side)."""
        return abs(float(notional)) * self.total_bps / 10_000.0
