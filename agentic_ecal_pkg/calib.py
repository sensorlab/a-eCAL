"""Measured two-rate calibration coefficients (paper Eqs. 4-5, fitted counterpart)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TwoRateCalib:
    """Two-rate coefficients (J/token) measured directly on the A100 sweeps.

    ``c_dec(b) = a_dec / b + kv_floor``. These are fitted, not derived: they are what a
    least-squares fit of the two-rate call model to the measured per-query energies returns, averaged over
    GPQA / GSM-Hard / MMLU-Hard under vLLM. They are the calibration counterpart of Eqs. (4)-(5),
    and the validation figure uses them so that the figure tests the two-rate *form* independently
    of the hardware derivation.

    Attributes:
        name (str): Human-readable model name.
        c_pre (float): Prefill coefficient, in Joules per prompt token.
        a_dec (float): Decode weight-read term, in Joules (amortised over the batch).
        kv_floor (float): Per-token KV-cache floor, in Joules, reached at large batch. Defaults
            to ``0.0``.
    """
    name: str
    c_pre: float
    a_dec: float
    kv_floor: float = 0.0

    def c_dec(self, batch: float) -> float:
        """Decode rate at a given serving batch.

        Args:
            batch (float): Serving batch size ``b`` (clamped to a minimum of 1).

        Returns:
            float: ``a_dec / max(b, 1) + kv_floor``, in Joules per generated token.
        """
        return self.a_dec / max(float(batch), 1.0) + self.kv_floor

    def energy(self, p_in: float, p_out: float, batch: float) -> float:
        """Energy of one call under the two-rate model with measured coefficients.

        Args:
            p_in (float): Prompt (prefill) tokens.
            p_out (float): Generated (decode) tokens.
            batch (float): Serving batch size ``b``.

        Returns:
            float: ``c_pre * p_in + c_dec(batch) * p_out``, in Joules.
        """
        return self.c_pre * p_in + self.c_dec(batch) * p_out
