from pathlib import Path
from typing import Optional

import numpy as np

from .common.data_models import (
    WiFiScenarioConfig,
    WiFiChannelRealization,
    WiFiTap,
    WiFiCluster,
    TapComponent,
)
from .parameters.parameter_loader import WiFiParameterLoader


class WiFiChannelSimulator:
    """Facade for WiFi channel generation (TGax-based)."""
    def __init__(self, config: WiFiScenarioConfig, parameter_file: Optional[str] = None):
        self.config = config
        if parameter_file is None:
            parameter_file = str(Path(__file__).resolve().parent / "parameters" / "wifi_tgax_models.json")
        self.params = WiFiParameterLoader.load_parameters(parameter_file)

    def _resolve_delay_scale(self) -> float:
        if self.config.delay_scale is not None:
            return float(self.config.delay_scale)
        bandwidth_hz = float(self.config.bandwidth_hz)
        if bandwidth_hz <= 40e6:
            return 1.0
        if bandwidth_hz <= 80e6:
            return 0.5
        if bandwidth_hz <= 160e6:
            return 0.25
        if bandwidth_hz <= 320e6:
            return 0.125
        return 0.125

    def _resolve_n_fft(self) -> int:
        if self.config.n_fft:
            return int(self.config.n_fft)
        subcarrier_spacing_hz = 312_500.0
        n_fft = int(round(self.config.bandwidth_hz / subcarrier_spacing_hz))
        return max(n_fft, 1)

    def _get_tap_phases(self, num_taps: int) -> np.ndarray:
        if self.config.phase_policy == "zero":
            return np.zeros(num_taps)
        rng = np.random.default_rng(self.config.random_seed)
        return rng.uniform(0.0, 2 * np.pi, num_taps)

    def _parse_model(self) -> tuple[list[WiFiTap], list[WiFiCluster], list[TapComponent]]:
        model_params = self.params.get_model(self.config.model_name)
        if not model_params:
            raise ValueError(f"WiFi model '{self.config.model_name}' not found in parameter file.")

        clusters: list[WiFiCluster] = []
        taps: list[WiFiTap] = []
        delay_scale = self._resolve_delay_scale()
        for cluster in model_params.get("clusters", []):
            cluster_id = int(cluster.get("cluster_id", 0))
            clusters.append(WiFiCluster(
                cluster_id=cluster_id,
                aoa_deg=float(cluster.get("aoa_deg", 0.0)),
                aod_deg=float(cluster.get("aod_deg", 0.0)),
                as_rx_deg=float(cluster.get("as_rx_deg", 0.0)),
                as_tx_deg=float(cluster.get("as_tx_deg", 0.0)),
            ))

            for tap in cluster.get("taps", []):
                delay_ns = float(tap.get("delay_ns", 0.0))
                power_db = float(tap.get("power_db", 0.0))
                taps.append(WiFiTap(
                    delay_s=delay_ns * 1e-9 * delay_scale,
                    power_linear=10 ** (power_db / 10.0),
                    cluster_id=cluster_id,
                ))

        if self.config.normalize_power:
            total_power = sum(tap.power_linear for tap in taps)
            if total_power > 0:
                for tap in taps:
                    tap.power_linear /= total_power

        cluster_map = {c.cluster_id: c for c in clusters}
        tap_components: list[TapComponent] = []
        for tap in taps:
            cluster = cluster_map.get(tap.cluster_id)
            if cluster is None:
                raise ValueError(f"Tap references unknown cluster_id={tap.cluster_id}.")
            tap_components.append(TapComponent(
                delay_s=tap.delay_s,
                power_linear=tap.power_linear,
                aoa_deg=cluster.aoa_deg,
                aod_deg=cluster.aod_deg,
                as_rx_deg=cluster.as_rx_deg,
                as_tx_deg=cluster.as_tx_deg,
            ))

        return taps, clusters, tap_components

    def run(self) -> WiFiChannelRealization:
        taps, clusters, tap_components = self._parse_model()
        tap_phases = self._get_tap_phases(len(taps))
        tap_coeffs = [
            np.sqrt(tap.power_linear) * np.exp(1j * phase)
            for tap, phase in zip(taps, tap_phases)
        ]
        impulse_response = [
            {
                "delay": tap.delay_s,
                "power_linear": tap.power_linear,
                "cluster_index": tap.cluster_id,
                "coeff": coeff,
            }
            for tap, coeff in zip(taps, tap_coeffs)
        ]
        h_subcarriers = None
        if self.config.enable_ofdm and taps:
            n_fft = self._resolve_n_fft()
            subcarrier_spacing_hz = self.config.bandwidth_hz / n_fft
            freq_offsets = (np.arange(n_fft) - n_fft / 2) * subcarrier_spacing_hz
            h_subcarriers = np.zeros((n_fft, 1, 1), dtype=complex)
            for tap, coeff in zip(taps, tap_coeffs):
                h_subcarriers[:, 0, 0] += coeff * np.exp(-1j * 2 * np.pi * freq_offsets * tap.delay_s)
        return WiFiChannelRealization(
            channel_matrix=None,
            impulse_response=impulse_response,
            taps=taps,
            clusters=clusters,
            tap_components=tap_components,
            h_subcarriers=h_subcarriers,
        )
