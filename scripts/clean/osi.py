#!/usr/bin/env python3
"""eCAL Eq. (3): the OSI-layer data-collection energy model, E_DC.

Chou et al. (arXiv:2408.00540v4), Eq. (3), sums over the OSI stack for each link p

    E_DC,p = P_T,p B_T,p / R_T,p                      PHY transmit
           + P_R,p B_T,p / R_R,p                      PHY receive
           + sum_{l=2}^{L} B_T,l,p ( N_dev,l P_dev + N_gw,l P_gw )   L2..L7 processing

with E_DC = sum_p E_DC,p over the links (Eq. 4). Three properties distinguish it from a lumped
per-bit intensity, and all three are reproduced here:

  * bits CASCADE. Each layer adds data- and control-plane overhead, and the inflated count is
    what the next layer down processes. Over the reference stack this is 2.35x by the PHY.
  * both directions are charged, sender and receiver.
  * per-layer PROCESSING at the endpoints is separate from the link cost, and on the reference
    configuration it is the majority of the total.

The reference implementation adds an expected-retransmission factor 1/(1-p), reproduced here.

COEFFICIENT PROVENANCE. The overhead ratios and per-layer processing energies below are the
defaults of the reference implementation (github.com/sensorlab/eCAL,
``src/ecal/configs/protocol_configs.py``, HTTP/TLS/RPC/TCP/IPv4/WIFI_MAC/WIFI_PHY), transcribed
unchanged. Two deliberate departures, both documented at their point of use in ``segment_energy``:

  1. ``N_dev`` and ``N_gw`` are set to 1 rather than 100. The reference value describes an
     aggregation scenario with 100 IoT nodes behind a gateway; an inter-agent hand-off is one
     sender and one receiver.
  2. The PHY term is not taken from the reference Wi-Fi profile but from the per-bearer intensity
     of the manuscript's own cited sources, which are network-level figures.

Consequently transit segments are charged the cascaded bit count at their cited intensity and are
NOT additionally charged L2..L7 processing: a network-level J/bit already includes the equipment
that performs it, and charging both would double count. Endpoint stack processing is charged once
per hand-off, for the two hosts that terminate it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

__all__ = ["Layer", "REFERENCE_STACK", "BYPASS_STACK", "cascade_bits",
           "endpoint_stack_energy", "segment_energy", "effective_intensity"]


@dataclass(frozen=True)
class Layer:
    """One OSI layer. Overheads are fractions of the bits arriving from the layer above."""
    name: str
    protocol: str
    data_plane_overhead: float
    control_plane_overhead: float
    energy_per_bit_sender: float      # J/bit
    energy_per_bit_receiver: float    # J/bit
    cycles_dev: float = 1.0           # N_dev in Eq. (3); 1 sender, not an aggregation of nodes
    power_dev: float = 2.0e-10        # P_dev [J/cycle]
    cycles_gw: float = 1.0            # N_gw
    power_gw: float = 1.0e-10         # P_gw [J/cycle]

    @property
    def processing_per_bit(self) -> float:
        return (self.energy_per_bit_sender + self.energy_per_bit_receiver
                + self.cycles_dev * self.power_dev + self.cycles_gw * self.power_gw)


# Transcribed from the reference implementation's defaults, top of stack first.
REFERENCE_STACK: Sequence[Layer] = (
    Layer("application",  "HTTP",     0.10, 0.05, 1.0e-08, 1.0e-08),
    Layer("presentation", "TLS",      0.08, 0.03, 2.0e-08, 2.0e-08),
    Layer("session",      "RPC",      0.02, 0.02, 1.0e-08, 1.0e-08),
    Layer("transport",    "TCP",      0.05, 0.10, 2.0e-08, 2.0e-08),
    Layer("network",      "IPv4",     0.03, 0.05, 2.0e-08, 2.0e-08),
    Layer("datalink",     "WIFI_MAC", 0.06, 0.08, 4.0e-08, 4.0e-08),
    Layer("physical",     "WIFI_PHY", 0.10, 0.15, 1.0e-07, 1.0e-07),
)


# Transport that bypasses the software stack: RDMA over converged Ethernet or InfiniBand, and
# NVLink inside a node, terminate framing in the adapter and never build L3..L7 headers in host
# memory. Only link and physical framing remain, and the endpoint processing term vanishes. This
# is how disaggregated prefill/decode deployments actually move KV cache between GPUs.
BYPASS_STACK: Sequence[Layer] = REFERENCE_STACK[-2:]


def cascade_bits(payload_bits: float, stack: Sequence[Layer] = REFERENCE_STACK) -> float:
    """Bits on the wire after every layer has added its overhead (Eq. 3's B_T,l cascade)."""
    bits = float(payload_bits)
    for layer in stack:
        bits += bits * (layer.data_plane_overhead + layer.control_plane_overhead)
    return bits


def endpoint_stack_energy(payload_bits: float, stack: Sequence[Layer] = REFERENCE_STACK,
                          include_phy: bool = False) -> float:
    """L2..L7 processing energy at the two hosts terminating a hand-off [J].

    The PHY layer is excluded by default: its cost is carried by the bearer intensity, which is a
    network-level figure. Set ``include_phy`` to recover the reference model exactly.
    """
    bits, total = float(payload_bits), 0.0
    for layer in stack:
        bits += bits * (layer.data_plane_overhead + layer.control_plane_overhead)
        if layer.name == "physical" and not include_phy:
            continue
        total += bits * layer.processing_per_bit
    return total


def segment_energy(payload_bits: float, eps: float, failure_rate: float = 0.0,
                   stack: Sequence[Layer] = REFERENCE_STACK) -> float:
    """Link energy [J] for one transit segment: cascaded bits at the bearer's intensity.

    No L2..L7 processing is added here -- see COEFFICIENT PROVENANCE above.
    """
    if not 0.0 <= failure_rate < 1.0:
        raise ValueError("failure_rate must be in [0, 1)")
    return cascade_bits(payload_bits, stack) * eps / (1.0 - failure_rate)


def effective_intensity(eps: float, endpoint: bool = True,
                        stack: Sequence[Layer] = REFERENCE_STACK) -> float:
    """Energy per PAYLOAD bit once Eq. (3)'s inflation and endpoint processing are included."""
    per_bit = cascade_bits(1.0, stack) * eps
    if endpoint:
        per_bit += endpoint_stack_energy(1.0, stack)
    return per_bit


if __name__ == "__main__":
    b = 21930.0
    print(f"payload {b:.0f} bits -> {cascade_bits(b):.0f} on the wire "
          f"({cascade_bits(b) / b:.2f}x inflation)")
    print(f"endpoint L2-L7 processing: {endpoint_stack_energy(b):.3e} J "
          f"({endpoint_stack_energy(1.0):.2e} J per payload bit)")
    print(f"\n{'bearer':<24}{'eps':>10}{'eps_eff':>11}{'ratio':>8}")
    for name, eps in (("backbone / metro fibre", 1e-8), ("FTTH access", 1e-7),
                      ("5G RAN", 1e-6), ("loaded edge cell", 1e-5), ("NB-IoT uplink", 1e-3)):
        eff = effective_intensity(eps)
        print(f"{name:<24}{eps:10.0e}{eff:11.2e}{eff / eps:7.0f}x")
