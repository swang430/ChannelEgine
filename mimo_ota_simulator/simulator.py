import numpy as np
from typing import Dict, Literal, List, Optional
import warnings

from .data_models import ChamberConfig, TargetChannelConfig, UniversalChannelProfile
from .channel_builders import create_builder
from channel_model_38901.simulator import ChannelSimulator
from channel_model_38901.common.data_models import EngineCluster
from channel_model_38901.common.geometry import apply_orientation
from wifi_channel_model.simulator import WiFiChannelSimulator
from wifi_channel_model.common.data_models import WiFiScenarioConfig

class MIMO_OTA_Simulator:
    def _calculate_probe_positions(self, chamber_config: ChamberConfig) -> np.ndarray:
        """Calculates the 3D Cartesian coordinates of each physical probe.

        P2-7: supports two layouts:
          - 'ring' (default, TR 37.977 §6.1 standard): uniform horizontal-plane
            ring computed from num_probes + radius_m alone.
          - 'custom' / 'multi-ring': explicit probe_positions list with per-probe
            (azimuth_deg, elevation_deg); radius_m applied uniformly.
            (MIMO-First Probe.position {az, el} maps 1:1 to this schema.)

        Both layouts return (num_probes, 3) Cartesian coordinates with the
        chamber center at origin. Single-pol mode in caller; dual-pol port
        doubling is downstream of this method.
        """
        num_probes = chamber_config.num_probes
        radius = chamber_config.radius_m

        if chamber_config.distribution == 'ring':
            angles = np.arange(num_probes) * (2 * np.pi / num_probes)
            x_coords = radius * np.cos(angles)
            y_coords = radius * np.sin(angles)
            z_coords = np.zeros(num_probes)
            return np.stack([x_coords, y_coords, z_coords], axis=1)

        if chamber_config.distribution in ('custom', 'multi-ring'):
            # Validator has ensured probe_positions is set and len == num_probes.
            az_rad = np.deg2rad([p.azimuth_deg for p in chamber_config.probe_positions])
            el_rad = np.deg2rad([p.elevation_deg for p in chamber_config.probe_positions])
            # Sphere → Cartesian. el=0 reduces exactly to ring formula (z=0).
            x_coords = radius * np.cos(el_rad) * np.cos(az_rad)
            y_coords = radius * np.cos(el_rad) * np.sin(az_rad)
            z_coords = radius * np.sin(el_rad)
            return np.stack([x_coords, y_coords, z_coords], axis=1)

        raise NotImplementedError(
            f"Probe distribution '{chamber_config.distribution}' is not supported."
        )

    @staticmethod
    def _calculate_physical_channel_properties(probe_position: np.ndarray, center_freq_hz: float) -> tuple[float, complex]:
        """Calculates the free-space path loss and phase factor for a single link."""
        speed_of_light = 299792458.0
        distance = np.linalg.norm(probe_position)
        if distance < 1e-9:
            return 0.0, 1.0
        wavelength = speed_of_light / center_freq_hz
        path_loss_db = 20 * np.log10(distance) + 20 * np.log10(center_freq_hz) - 147.55
        phase_factor = np.exp(-1j * 2 * np.pi * distance / wavelength)
        return path_loss_db, phase_factor

    def _synthesize_b_matrix(self, chamber_config: ChamberConfig, target_config: TargetChannelConfig,
                              frequency_hz_override: Optional[float] = None) -> np.ndarray:
        """Synthesizes the B-matrix based on physical properties.

        ``frequency_hz_override`` lets the caller force the working frequency
        (probe geometry phase computation + free-space path loss). This is
        required when ``input_mode='custom'`` because ``CustomCDLProfile``
        is the source of truth for frequency, and ``TargetChannelConfig.
        center_frequency_hz`` may still be at its 2.4 GHz default.

        Phase 6 dual-pol behavior (chamber.dual_polarized=True):
          - Each physical probe has 2 ports: V (index 2p, θ-polarized) and
            H (index 2p+1, φ-polarized).
          - DUT antenna polarization (target_config.rx_antenna.polarization)
            picks which probe port couples; the orthogonal port gets 0.
          - rx_antenna.polarization='V' → V ports nonzero, H ports zero
          - rx_antenna.polarization='H' → H ports nonzero, V ports zero
          - (slant pol / per-antenna pol are Phase 7, deferred)

        Single-pol mode (chamber.dual_polarized=False) keeps prior behavior
        and implicitly assumes probe/DUT polarization match. Explicit
        rx_antenna.polarization='H' under single-pol mode raises ValueError
        (per OQ4: 'H' single-pol is meaningless config; force caller to
        set dual_polarized=True or polarization='V').
        """
        rx_pol = target_config.rx_antenna.polarization
        if not chamber_config.dual_polarized and rx_pol != 'V':
            raise ValueError(
                f"chamber.dual_polarized=False but rx_antenna.polarization="
                f"{rx_pol!r}. Single-pol mode implicitly assumes matched "
                f"polarization (V); any non-V polarization ('H', '+45', '-45') "
                f"requires chamber.dual_polarized=True. Either set "
                f"dual_polarized=True or polarization='V'."
            )

        num_rx_antennas = target_config.rx_antenna.num_rows * target_config.rx_antenna.num_cols
        num_effective_probes = chamber_config.num_probes * (2 if chamber_config.dual_polarized else 1)
        center_freq_hz = (
            frequency_hz_override
            if frequency_hz_override is not None
            else target_config.center_frequency_hz
        )
        speed_of_light = 299792458.0
        wavelength = speed_of_light / center_freq_hz

        b_matrix = np.zeros((num_rx_antennas, num_effective_probes), dtype=complex)

        probe_positions = self._calculate_probe_positions(chamber_config)
        dut_antenna_positions = self._calculate_dut_antenna_positions(target_config.rx_antenna, wavelength)

        # Rx-side polarization projection for dual-pol mode. Convention:
        # 2p = V port (θ), 2p+1 = H port (φ). The DUT couples to each port by the
        # corresponding component of its [θ, φ] pol vector [cos ζ, sin ζ], so V/H
        # remain port-exclusive while ±45° slant couples to both ports by cos45°.
        rx_pol_vec = self._pol_vec(rx_pol)   # (2,) -> [θ-weight, φ-weight]

        for p_idx in range(chamber_config.num_probes):
            probe_pos = probe_positions[p_idx]
            for r_idx in range(num_rx_antennas):
                rx_ant_pos = dut_antenna_positions[r_idx]
                distance_vec = probe_pos - rx_ant_pos

                path_loss_db, phase_factor = self._calculate_physical_channel_properties(
                    distance_vec, center_freq_hz
                )
                path_gain_linear = 10**(-path_loss_db / 20.0)

                # Incorporate probe antenna gain
                probe_gain_linear = 10**(chamber_config.probe_antenna.gain_dBi / 20.0)
                rx_gain_linear = 1.0 # Assume isotropic DUT antenna for now

                polarization_factor = 1.0

                channel_response = path_gain_linear * phase_factor * probe_gain_linear * rx_gain_linear * polarization_factor

                if not chamber_config.dual_polarized:
                    b_matrix[r_idx, p_idx] = channel_response
                else:
                    # Dual-pol: project the channel onto V/H probe ports by the
                    # DUT pol vector. V→[1,0], H→[0,1] stay port-exclusive (legacy
                    # bit-identical); ±45° couples to both ports by cos45°.
                    b_matrix[r_idx, 2*p_idx]     = channel_response * rx_pol_vec[0]
                    b_matrix[r_idx, 2*p_idx + 1] = channel_response * rx_pol_vec[1]

        # Apply calibration vector if provided
        if chamber_config.calibration_vector is not None:
            if chamber_config.calibration_vector.shape[0] == num_effective_probes:
                b_matrix *= chamber_config.calibration_vector
            else:
                warnings.warn("Calibration vector shape mismatch. Ignoring calibration.")

        return b_matrix

    @staticmethod
    def _calculate_dut_antenna_positions(antenna_config, wavelength: float) -> np.ndarray:
        """Calculates the 3D Cartesian coordinates of each DUT antenna element."""
        num_rows = antenna_config.num_rows
        num_cols = antenna_config.num_cols
        spacing_h = antenna_config.spacing_h * wavelength
        spacing_v = antenna_config.spacing_v * wavelength

        if antenna_config.array_type == 'ULA':
            x_coords = np.linspace(-0.5 * (num_cols - 1) * spacing_h, 0.5 * (num_cols - 1) * spacing_h, num_cols)
            y_coords = np.zeros(num_cols)
            z_coords = np.zeros(num_cols)
            positions = np.stack([x_coords, y_coords, z_coords], axis=1)

        elif antenna_config.array_type == 'URA':
            x_coords = np.linspace(-0.5 * (num_cols - 1) * spacing_h, 0.5 * (num_cols - 1) * spacing_h, num_cols)
            z_coords = np.linspace(-0.5 * (num_rows - 1) * spacing_v, 0.5 * (num_rows - 1) * spacing_v, num_rows)
            xx, zz = np.meshgrid(x_coords, z_coords)
            yy = np.zeros_like(xx)
            positions = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)

        else:
            raise NotImplementedError(f"DUT antenna array type '{antenna_config.array_type}' is not yet supported.")

        # Apply the DUT's fixed orientation in the chamber (TR 38.901 Sec 7.1.3).
        # Default 0 => identity, so legacy / 360°-rotation tests are bit-identical.
        return apply_orientation(
            positions,
            getattr(antenna_config, 'bearing_deg', 0.0),
            getattr(antenna_config, 'downtilt_deg', 0.0),
            getattr(antenna_config, 'slant_deg', 0.0),
        )

    # Default XPR when caller does not set cluster.xpr_linear (Phase 6 OQ2):
    # 7 dB matches TR 38.901 §7.7-2 typical UMa/UMi NLOS cross-polarization ratio.
    DEFAULT_XPR_DB = 7.0
    _DEFAULT_XPR_LINEAR = 10 ** 0.7  # ≈ 5.0119

    def _build_pol_matrix(self, cluster: EngineCluster) -> np.ndarray:
        """Build the 2×2 per-cluster polarization mixing matrix per TR 38.901 §7.3.2.

        Returns a complex (2, 2) array:
            [[exp(jΦ_θθ),             sqrt(1/κ)·exp(jΦ_θφ)],
             [sqrt(1/κ)·exp(jΦ_φθ),   exp(jΦ_φφ)         ]]

        where κ = XPR_linear and Φ_* are per-cluster initial phases.

        Falls back to defaults when fields are missing:
          - xpr_linear missing → DEFAULT_XPR_DB (7 dB) + one-time warning
          - initial_phases missing → all zeros (no phase shift)

        Phase 6 OQ2: warn (not raise) so legacy callers that have dual_polarized=True
        but never supplied xpr_db still produce a sensible (rather than ZeroDivision)
        result; explicit xpr_db on the cluster suppresses the warning.
        """
        if cluster.xpr_linear is not None:
            kappa = float(cluster.xpr_linear)
            if kappa <= 0:
                raise ValueError(
                    f"cluster.xpr_linear must be positive; got {kappa}"
                )
        else:
            warnings.warn(
                f"EngineCluster.xpr_linear is None under dual-pol synthesis; "
                f"falling back to default {self.DEFAULT_XPR_DB} dB "
                f"(TR 38.901 §7.7-2 typical NLOS value). Set "
                f"CDLClusterSpec.xpr_db explicitly to suppress this warning.",
                stacklevel=2,
            )
            kappa = self._DEFAULT_XPR_LINEAR

        if cluster.initial_phases is not None and len(cluster.initial_phases) >= 4:
            phi = np.asarray(cluster.initial_phases[:4], dtype=float)
        else:
            phi = np.zeros(4, dtype=float)

        sqrt_inv_kappa = 1.0 / np.sqrt(kappa)
        return np.array([
            [np.exp(1j * phi[0]),                  sqrt_inv_kappa * np.exp(1j * phi[1])],
            [sqrt_inv_kappa * np.exp(1j * phi[2]), np.exp(1j * phi[3])],
        ], dtype=complex)

    # [θ-component, φ-component] basis vector [cos ζ, sin ζ] per TR 38.901 §7.3.2,
    # for slant angle ζ: V=0°, H=90°, ±45° slant. Stored exactly so V/H keep their
    # hard zeros (no cos(90°)≈6e-17 leakage into the orthogonal probe port).
    _INV_SQRT2 = float(1.0 / np.sqrt(2.0))
    _POL_VECS = {
        'V': (1.0, 0.0),
        'H': (0.0, 1.0),
        '+45': (_INV_SQRT2, _INV_SQRT2),
        '-45': (_INV_SQRT2, -_INV_SQRT2),
    }

    @classmethod
    def _pol_vec(cls, polarization: str) -> np.ndarray:
        """Return the [θ-component, φ-component] basis vector for a polarization label."""
        if polarization not in cls._POL_VECS:
            raise ValueError(
                f"Unsupported polarization {polarization!r}; supported: "
                f"{sorted(cls._POL_VECS)}."
            )
        return np.array(cls._POL_VECS[polarization])

    @staticmethod
    def _estimate_pathloss_db(bs_position, ue_position, center_freq_hz: float) -> float:
        """Estimates free-space path loss between BS and UE."""
        bs = np.array(bs_position, dtype=float)
        ue = np.array(ue_position, dtype=float)
        distance = np.linalg.norm(bs - ue)
        if distance < 1e-9:
            return 0.0
        return 20 * np.log10(distance) + 20 * np.log10(center_freq_hz) - 147.55

    @staticmethod
    def _unit_vector_from_angles(azimuth_deg: float, zenith_deg: float = 90.0) -> np.ndarray:
        theta_rad = np.deg2rad(zenith_deg)
        phi_rad = np.deg2rad(azimuth_deg)
        return np.array([
            np.sin(theta_rad) * np.cos(phi_rad),
            np.sin(theta_rad) * np.sin(phi_rad),
            np.cos(theta_rad)
        ])

    @staticmethod
    def _array_response(positions: np.ndarray, unit_vec: np.ndarray, k_factor: float) -> np.ndarray:
        return np.exp(1j * k_factor * (positions @ unit_vec))

    @staticmethod
    def _apply_reciprocity_transform(matrix: np.ndarray, mode: str) -> np.ndarray:
        if mode == "transpose":
            return matrix.T
        if mode == "conjugate_transpose":
            return matrix.conj().T
        raise ValueError(f"Unknown reciprocity_transform: {mode}")

    def _build_ul_impulse_response(self, impulse_response: List[Dict], mode: str) -> List[Dict]:
        ul_response: List[Dict] = []
        for tap in impulse_response:
            coeff = tap.get("coeff")
            if coeff is None:
                continue
            coeff_ul = self._apply_reciprocity_transform(coeff, mode)
            ul_response.append({
                "delay": tap["delay"],
                "coeff": coeff_ul,
                "cluster_index": tap.get("cluster_index", 0),
            })
        return ul_response

    def _build_ul_ray_components(self, ray_components: Optional[List[Dict]], mode: str) -> Optional[List[Dict]]:
        if not ray_components:
            return None
        # UL mirror via reciprocity: W_base is transposed (P, Tx) -> (Tx, P), so
        # the probe-indexed phi_init vector now aligns with COLUMNS, not rows.
        # We flip phi_axis here so downstream code (e.g. exporter broadcast) is
        # explicit and doesn't infer axis from shape — that inference is wrong
        # when num_tx == num_probes (square W_base).
        dl_axis = ray_components[0].get('phi_axis', 'row')
        ul_axis_map = {'row': 'col', 'col': 'row'}
        ul_axis = ul_axis_map.get(dl_axis, 'col')
        ul_components = []
        for rc in ray_components:
            w_base = rc["W_base"]
            phi_init = rc["phi_init"]
            doppler_hz = rc["doppler_hz"]
            if mode == "transpose":
                w_ul = w_base.T
                phi_ul = phi_init
                doppler_ul = doppler_hz
            elif mode == "conjugate_transpose":
                w_ul = w_base.conj().T
                phi_ul = -phi_init
                doppler_ul = -doppler_hz
            else:
                raise ValueError(f"Unknown reciprocity_transform: {mode}")
            ul_components.append({
                "W_base": w_ul,
                "phi_init": phi_ul,
                "doppler_hz": doppler_ul,
                "cluster_index": rc.get("cluster_index", 0),
                "phi_axis": ul_axis,
            })
        return ul_components

    def _synthesize_wifi_clusters(
        self,
        tap_components,
        tx_positions: np.ndarray,
        rx_positions: np.ndarray,
        wavelength: float,
        phase_policy: str,
        random_seed: Optional[int],
    ) -> tuple[list[EngineCluster], np.ndarray]:
        """Builds EngineCluster list and a narrowband channel matrix from WiFi tap components."""
        num_rx = rx_positions.shape[0]
        num_tx = tx_positions.shape[0]
        k_factor = 2 * np.pi / wavelength
        rng = np.random.default_rng(random_seed)
        target_h = np.zeros((num_rx, num_tx), dtype=complex)
        clusters: list[EngineCluster] = []

        for tap in tap_components or []:
            cluster = EngineCluster(
                delay=tap.delay_s,
                power=tap.power_linear,
                aod_phi=tap.aod_deg,
                aoa_phi=tap.aoa_deg,
                zod_theta=90.0,
                zoa_theta=90.0,
                as_aod=tap.as_tx_deg,
                as_aoa=tap.as_rx_deg,
                as_zod=0.0,
                as_zoa=0.0,
            )
            phase_init = 0.0
            if phase_policy == "random":
                phase_init = rng.uniform(-np.pi, np.pi)
            k_hat_rx = self._unit_vector_from_angles(cluster.aoa_phi, cluster.zoa_theta)
            k_hat_tx = self._unit_vector_from_angles(cluster.aod_phi, cluster.zod_theta)
            a_rx = self._array_response(rx_positions, k_hat_rx, k_factor)
            a_tx = self._array_response(tx_positions, k_hat_tx, k_factor)
            cluster.h_matrix = np.outer(a_rx, a_tx) * np.exp(1j * phase_init)
            clusters.append(cluster)
            target_h += np.sqrt(cluster.power) * cluster.h_matrix

        return clusters, target_h

    @staticmethod
    def _calculate_weights_for_cluster(cluster: EngineCluster, chamber_config: ChamberConfig, num_candidate_probes: int = 4) -> np.ndarray:
        """Calculates probe weights for a single cluster to synthesize its AoA and AS.

        P2-7: when chamber_config has explicit `probe_positions` (non-ring layouts —
        sparse, sector, dual-ring), use the per-probe azimuth from the position
        list instead of the ring `(p-1)·2π/N` assumption. This is the core fix
        for the "ring-only silent constraint" flagged in MIMO-First
        docs/architecture/channel-engine-data-flow.md (2026-05-19).

        Note: weights are still azimuth-only; elevation is not consumed by the
        Gaussian here. ZoA-aware weighting is a potential P3 extension.
        """
        num_physical_probes = chamber_config.num_probes
        num_effective_probes = num_physical_probes * 2 if chamber_config.dual_polarized else num_physical_probes

        # P2-7: real azimuths when probe_positions is set, ring-uniform otherwise.
        if chamber_config.probe_positions is not None:
            probe_angles_rad = np.deg2rad(
                np.array([p.azimuth_deg for p in chamber_config.probe_positions])
            )
        else:
            probe_angles_rad = np.linspace(0, 2 * np.pi, num_physical_probes, endpoint=False)

        target_angle_rad = np.deg2rad(cluster.aoa_phi)
        target_as_rad = np.deg2rad(cluster.as_aoa)

        # --- ADAPTIVE BROADENING FIX ---
        # Ensure the Gaussian width is at least a fraction of the probe spacing.
        # For non-uniform layouts we use 2π/N as the "mean effective spacing" —
        # OK for sparse/sector approximations; future P3 work may switch to
        # per-probe nearest-neighbor distance for very irregular layouts.
        probe_spacing_rad = 2 * np.pi / num_physical_probes
        min_as_rad = probe_spacing_rad / 1.5  # Sigma approx 2/3 of spacing ensures overlap

        effective_as_rad = max(target_as_rad, min_as_rad)
        # -------------------------------

        angle_diffs = np.abs(probe_angles_rad - target_angle_rad)
        angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)
        candidate_indices = np.argsort(angle_diffs)[:num_candidate_probes]
        weights = np.zeros(num_physical_probes)
        
        for idx in candidate_indices:
            delta_angle = np.abs(probe_angles_rad[idx] - target_angle_rad)
            delta_angle = np.minimum(delta_angle, 2 * np.pi - delta_angle)
            
            # Use effective_as_rad for Gaussian
            weights[idx] = np.exp(-np.power(delta_angle, 2) / (2 * np.power(effective_as_rad, 2)))
            
        total_weight = np.sum(weights)
        if total_weight > 1e-9:
            weights /= total_weight
        final_weights = np.zeros((num_effective_probes, 1), dtype=float)
        if chamber_config.dual_polarized:
            for i in range(num_physical_probes):
                final_weights[i * 2, 0] = weights[i]
                final_weights[i * 2 + 1, 0] = weights[i]
        else:
            final_weights[:, 0] = weights
        return final_weights

    @staticmethod
    def _get_g_tx_from_h_matrix(h_matrix: np.ndarray) -> np.ndarray:
        """Extracts the transmit-side channel vector from a cluster's H matrix."""
        return h_matrix[0:1, :]

    def run(self, chamber_config: ChamberConfig, channel_config: TargetChannelConfig,
            synthesis_method: Literal['pinv', 'cluster', 'ray', 'strict_pfs'] = 'ray',
            probe_phase_jitter: bool = False) -> Dict:
        """Legacy entry point, delegates to builder and synthesize_ota."""
        if channel_config.standard == 'wifi':
            # Keep wifi legacy path intact for now (it seeds at _run_wifi L452-453)
            return self._run_wifi(chamber_config, channel_config, synthesis_method, probe_phase_jitter)

        # 3GPP path: honor channel_config.random_seed by seeding np.random ONCE
        # before delegating to the builder + OTA synthesis. Previously only the
        # WiFi branch did this, so 3GPP callers saw non-reproducible LSP/cluster
        # generation despite passing random_seed (the field was stored but unread).
        # GUI parity bug found 2026-05-18: gui.py compensated by np.random.seed()
        # before sim.run(); app.py did not, hence Streamlit users saw drifting LSPs.
        # Backward compat: random_seed=None → no auto-seeding (caller-controlled).
        if channel_config.random_seed is not None:
            np.random.seed(channel_config.random_seed)

        profile = create_builder(channel_config).build()
        return self.synthesize_ota(chamber_config, channel_config, profile, synthesis_method, probe_phase_jitter)

    def _run_wifi(self, chamber_config: ChamberConfig, channel_config: TargetChannelConfig, 
            synthesis_method: Literal['pinv', 'cluster', 'ray', 'strict_pfs'] = 'ray',
            probe_phase_jitter: bool = False) -> Dict:
        num_tx_antennas = channel_config.tx_antenna.num_cols * channel_config.tx_antenna.num_rows
        num_effective_probes = chamber_config.num_probes * 2 if chamber_config.dual_polarized else chamber_config.num_probes
        probe_tx_power_contributions = np.zeros((num_effective_probes, num_tx_antennas))
        cluster_probe_mapping = {}
        ray_components = None
        jitter_vector = None
        impulse_response = []
        if channel_config.random_seed is not None:
            np.random.seed(channel_config.random_seed)
        B_matrix = self._synthesize_b_matrix(chamber_config, channel_config)
        
        wavelength = 3e8 / channel_config.center_frequency_hz
        tx_positions = self._calculate_dut_antenna_positions(channel_config.tx_antenna, wavelength)
        rx_positions = self._calculate_dut_antenna_positions(channel_config.rx_antenna, wavelength)
        wifi_config = WiFiScenarioConfig(
            model_name=channel_config.model_name, center_frequency_hz=channel_config.center_frequency_hz,
            bandwidth_hz=channel_config.bandwidth_hz, enable_ofdm=channel_config.enable_ofdm, n_fft=channel_config.n_fft,
            sample_rate_hz=None, random_seed=channel_config.random_seed, phase_policy=channel_config.phase_policy,
            normalize_power=channel_config.normalize_power, delay_scale=channel_config.delay_scale,
            bs_positions=np.array([channel_config.bs_position]), ue_positions=np.array([channel_config.ue_position]),
            ue_velocities=np.array([channel_config.ue_velocity]),
        )
        wifi_simulator = WiFiChannelSimulator(wifi_config)
        wifi_result = wifi_simulator.run()
        physical_clusters, target_H_matrix = self._synthesize_wifi_clusters(
            wifi_result.tap_components, tx_positions, rx_positions, wavelength,
            wifi_config.phase_policy, wifi_config.random_seed,
        )
        lsps = None
        pathloss_db = self._estimate_pathloss_db(channel_config.bs_position, channel_config.ue_position, channel_config.center_frequency_hz)
        is_los = None
        
        profile = UniversalChannelProfile(
            center_frequency_hz=channel_config.center_frequency_hz,
            ue_velocity=np.array(channel_config.ue_velocity),
            pathloss_db=pathloss_db,
            clusters=physical_clusters,
            lsps=lsps,
            is_los=False
        )
        return self.synthesize_ota(chamber_config, channel_config, profile, synthesis_method, probe_phase_jitter, B_matrix, target_H_matrix=target_H_matrix)

    def synthesize_ota(self, chamber_config: ChamberConfig, channel_config: TargetChannelConfig, 
            channel_profile: UniversalChannelProfile,
            synthesis_method: Literal['pinv', 'cluster', 'ray', 'strict_pfs'] = 'ray',
            probe_phase_jitter: bool = False,
            B_matrix: Optional[np.ndarray] = None,
            target_H_matrix: Optional[np.ndarray] = None) -> Dict:
        num_tx_antennas = channel_config.tx_antenna.num_cols * channel_config.tx_antenna.num_rows
        num_effective_probes = chamber_config.num_probes * 2 if chamber_config.dual_polarized else chamber_config.num_probes
        probe_tx_power_contributions = np.zeros((num_effective_probes, num_tx_antennas))
        cluster_probe_mapping = {}
        ray_components = None # Initialize for export
        jitter_vector = None
        
        # Wideband Impulse Response: List of taps {delay, coeff_matrix (PxT)}
        impulse_response = []
        
        # synthesize_ota ignores random seed setting here since generation is already done

        # Detect frequency mismatch (Codex P2 on PR #5): when input_mode='custom'
        # caller sets profile.center_frequency_hz as the source of truth, but
        # channel_config.center_frequency_hz may still hold its 2.4 GHz default.
        # The profile wins (it's the actual channel description), but warn the
        # caller in case they intended to set both consistently.
        if (channel_config.input_mode == 'custom'
                and channel_config.center_frequency_hz != channel_profile.center_frequency_hz):
            warnings.warn(
                f"channel_config.center_frequency_hz "
                f"({channel_config.center_frequency_hz:.3e}) differs from "
                f"channel_profile.center_frequency_hz "
                f"({channel_profile.center_frequency_hz:.3e}); profile wins "
                f"(used for both B-matrix and synthesis). When input_mode='custom', "
                f"prefer setting frequency on the CustomCDLProfile only.",
                stacklevel=2,
            )

        if B_matrix is None:
            # Synthesize the physical B-Matrix using the profile's frequency
            # (the authoritative source post-build). See warning above.
            B_matrix = self._synthesize_b_matrix(
                chamber_config, channel_config,
                frequency_hz_override=channel_profile.center_frequency_hz,
            )

        wavelength = 3e8 / channel_profile.center_frequency_hz
        tx_positions = self._calculate_dut_antenna_positions(channel_config.tx_antenna, wavelength)
        rx_positions = self._calculate_dut_antenna_positions(channel_config.rx_antenna, wavelength)
        
        physical_clusters = channel_profile.clusters
        lsps = channel_profile.lsps
        pathloss_db = channel_profile.pathloss_db
        is_los = channel_profile.is_los
        scenario_params = None  # We assume Ray mapping offset parameters are standard or handled properly
        
        # If target_H_matrix is needed for pinv but missing, synthesize a dummy one
        if synthesis_method == 'pinv' and target_H_matrix is None:
            # Reconstruct H_matrix quickly
            num_rx_antennas = channel_config.rx_antenna.num_rows * channel_config.rx_antenna.num_cols
            target_H_matrix = np.zeros((num_rx_antennas, num_tx_antennas), dtype=complex)
            k_factor = 2 * np.pi / wavelength
            for c in physical_clusters:
                k_hat_rx = self._unit_vector_from_angles(c.aoa_phi, c.zoa_theta)
                k_hat_tx = self._unit_vector_from_angles(c.aod_phi, c.zod_theta)
                a_rx = self._array_response(rx_positions, k_hat_rx, k_factor)
                a_tx = self._array_response(tx_positions, k_hat_tx, k_factor)
                target_H_matrix += np.sqrt(c.power) * np.outer(a_rx, a_tx)

        # ---------------------------------------------

        if synthesis_method == 'pinv':
            warnings.warn(
                "synthesis_method='pinv' is deprecated and will be removed in a future release. "
                "It produces no wideband impulse response (narrowband-only) and is broken for "
                "Propsim F64 export. Use 'ray' for current production or wait for 'strict_pfs' "
                "(Phase 1 of strict-pfs-rollout) for the recommended replacement.",
                DeprecationWarning,
                stacklevel=2,
            )
            B_pinv = np.linalg.pinv(B_matrix)
            transmit_mapping_matrix_A = B_pinv @ target_H_matrix
            probe_tx_power_contributions = np.abs(transmit_mapping_matrix_A)**2

        elif synthesis_method == 'cluster':
            warnings.warn(
                "synthesis_method='cluster' is deprecated and will be removed in a future release. "
                "The rank-1 outer-product structure cannot represent strict PFS. "
                "Use 'ray' for current production or wait for 'strict_pfs' "
                "(Phase 1 of strict-pfs-rollout) for the recommended replacement.",
                DeprecationWarning,
                stacklevel=2,
            )
            if not physical_clusters or physical_clusters[0].h_matrix is None:
                raise ValueError("Cluster synthesis requires h_matrix in clusters. Engine not updated?")
            
            A_total = np.zeros((num_effective_probes, num_tx_antennas), dtype=complex)

            for i, cluster in enumerate(physical_clusters):
                w_map = self._calculate_weights_for_cluster(cluster, chamber_config)
                g_tx = self._get_g_tx_from_h_matrix(cluster.h_matrix)
                
                # FIX: Power is already linear from ChannelGenerator
                cluster_power_linear = cluster.power 
                
                A_cluster = np.sqrt(cluster_power_linear) * (w_map @ g_tx)
                A_total += A_cluster
                
                # Store Wideband Tap
                impulse_response.append({
                    'delay': cluster.delay,
                    'coeff': A_cluster, # (PxT)
                    'cluster_index': i
                })

                power_contributions = (w_map * cluster_power_linear).flatten()
                cluster_probe_mapping[i] = power_contributions

            transmit_mapping_matrix_A = A_total
            probe_tx_power_contributions = np.abs(transmit_mapping_matrix_A)**2
            
        elif synthesis_method in ('ray', 'strict_pfs'):
            # Ray-Level Synthesis (Geometric GSCM)
            # This reconstructs rays from cluster parameters, bypassing the 'Rank-1' issue of simplified models.
            #
            # Two modes share this branch (per design §2 / §3):
            #   - 'ray':        phi_rnd is a SCALAR per ray, shared across probes.
            #                   Cross-probe coherence is geometry-driven; strict PFS only
            #                   holds in the ensemble-average limit and benefits from jitter.
            #   - 'strict_pfs': phi_rnd is a (N_probes,) VECTOR with iid U[-pi, pi] entries.
            #                   E[A_i * conj(A_j)] = 0 for i != j per-realization (true strict PFS).
            #                   Use case: avoid output phase calibration in hardware loop.
            use_strict_pfs = (synthesis_method == 'strict_pfs')

            # Phase 6 T6.4: dual-pol synthesis gating.
            # When chamber.dual_polarized=True, each cluster gets a 2×2 polarization
            # mixing matrix M = _build_pol_matrix(cluster). We project ONLY on the
            # Tx side here (m_col = M @ tx_pol_vec), producing two complex factors:
            #   m_col[0] for V probe ports (drives the θ-component of the channel)
            #   m_col[1] for H probe ports (drives the φ-component)
            # Rx-side projection lives in _synthesize_b_matrix (T6.2): the B-matrix
            # zeros out the H ports for V-pol DUT and vice versa. Composing
            # effective_H = B · A then reconstructs the full
            #   rx_pol_vec · M · tx_pol_vec
            # scalar per cluster as required by TR 38.901 §7.3.2.
            #
            # Why split this way: the A-matrix output ('transmit_mapping_matrix_A'
            # and the derived 'probe_tx_power_contributions') is exposed to users
            # for diagnostics. It should reflect what a real chamber driver would
            # write to each probe port — including the H drive that a V DUT can't
            # receive — so probe-power monitoring stays physically meaningful.
            # Single-pol mode keeps pol_scalar = 1 (legacy bit-identical).
            use_dual_pol = bool(chamber_config.dual_polarized)
            if use_dual_pol:
                tx_pol_vec = self._pol_vec(channel_config.tx_antenna.polarization)

            # 1. Get Ray Offsets
            default_ray_offsets = np.array([
                0.0447, -0.0447, 0.1413, -0.1413, 0.2492, -0.2492, 0.3715, -0.3715,
                0.5129, -0.5129, 0.6797, -0.6797, 0.8844, -0.8844, 1.1481, -1.1481,
                1.5195, -1.5195, 2.1551, -2.1551
            ])
            if scenario_params and 'ray_angle_offsets_deg' in scenario_params:
                ray_offsets = np.array(scenario_params['ray_angle_offsets_deg'])
            else:
                ray_offsets = default_ray_offsets
            
            A_total = np.zeros((num_effective_probes, num_tx_antennas), dtype=complex)
            c = 3e8
            lambda_0 = c / channel_profile.center_frequency_hz
            k_factor = 2 * np.pi / lambda_0
            
            # Store ray components for time-series generation (Doppler)
            ray_components = []
            
            for i, cluster in enumerate(physical_clusters):
                # We will map each ray to probes independently
                cluster_power_accum = np.zeros(num_effective_probes)
                cluster_A_accum = np.zeros((num_effective_probes, num_tx_antennas), dtype=complex)

                # Phase 6 T6.4: per-cluster Tx-side polarization projection.
                # Build the 2×2 polarization matrix once per cluster (invariant
                # across sub-rays) then project onto the Tx pol basis. Result is
                # a (2,) complex vector: m_col[0] for V probe ports, m_col[1] for
                # H probe ports. Single-pol mode uses a uniform 1.0 factor.
                if use_dual_pol:
                    pol_matrix = self._build_pol_matrix(cluster)  # (2, 2)
                    m_col = pol_matrix @ tx_pol_vec  # (2,) complex
                    # Expand to (num_effective_probes, 1) for broadcast against W_base.
                    # Even indices (2p) = V port → m_col[0]; odd (2p+1) = H port → m_col[1].
                    pol_factor_per_port = np.empty(num_effective_probes, dtype=complex)
                    pol_factor_per_port[0::2] = m_col[0]
                    pol_factor_per_port[1::2] = m_col[1]
                    pol_factor_col = pol_factor_per_port[:, None]   # (P_eff, 1)
                else:
                    pol_factor_col = 1.0 + 0j  # broadcasts as scalar in W_base mul

                for m in range(20):
                    alpha = ray_offsets[m]
                    c_scale = 1.0 # Scaling factor
                    
                    # 1. Calculate Ray Angles
                    aoa_ray = cluster.aoa_phi + c_scale * alpha * cluster.as_aoa
                    aod_ray = cluster.aod_phi + c_scale * alpha * cluster.as_aod
                    zoa_ray = cluster.zoa_theta + c_scale * alpha * cluster.as_zoa
                    zod_ray = cluster.zod_theta + c_scale * alpha * cluster.as_zod
                    
                    # 2. Calculate Probe Weights for this Ray (Point Source Mode)
                    # We treat each ray as a point source. 
                    # CRITICAL FIX: Do NOT use AS=0 (Nearest Neighbor). This causes phase quantization.
                    # Use a small AS (e.g., 2 deg) to enable Gaussian Interpolation between probes.
                    # This smooths the field and improves spatial accuracy.
                    temp_ray_cluster = EngineCluster(aod_phi=aod_ray, aoa_phi=aoa_ray, power=0, delay=0, as_aod=0, as_aoa=2.0)
                    w_map_ray = self._calculate_weights_for_cluster(temp_ray_cluster, chamber_config) # Shape (N_probes, 1)
                    
                    # 3. Calculate Tx Steering Vector (g_tx for this ray)
                    # v_tx = exp(j * k * r_tx . k_hat_departure)
                    theta_t_rad = np.deg2rad(zod_ray)
                    phi_t_rad = np.deg2rad(aod_ray)
                    k_hat_tx = np.array([
                        np.sin(theta_t_rad) * np.cos(phi_t_rad),
                        np.sin(theta_t_rad) * np.sin(phi_t_rad),
                        np.cos(theta_t_rad)
                    ])
                    
                    g_tx_ray = np.zeros((1, num_tx_antennas), dtype=complex)
                    for tx_idx in range(num_tx_antennas):
                        pos_tx = tx_positions[tx_idx]
                        phase_tx = k_factor * np.dot(k_hat_tx, pos_tx)
                        g_tx_ray[0, tx_idx] = np.exp(1j * phase_tx)
                    
                    # 4. Synthesize Contribution
                    # Random phase for this ray (Decorrelation source!)
                    # ray mode: scalar (broadcasts to all probes — geometry-coupled).
                    # strict_pfs mode: (N_probes,) iid vector — per-probe independent fading.
                    #
                    # Phase 5 T5.5: if the cluster carries CDL initial_phases
                    # (4-element polarization phases [Θθθ, Θθφ, Θφθ, Θφφ]),
                    # use index 0 (Θθθ) as a per-cluster base phase shared by
                    # all sub-rays. Indices 1-3 (cross-polarization) are plumbed
                    # but not consumed until dual-pol synthesizer (Phase 6).
                    # The base + uniform construction preserves strict_pfs
                    # cross-correlation contract: E[exp(j(base + rnd_i - base - rnd_j))]
                    # = E[exp(j(rnd_i - rnd_j))] = 0 for i ≠ j.
                    base_phase = (
                        float(cluster.initial_phases[0])
                        if cluster.initial_phases is not None and len(cluster.initial_phases) >= 1
                        else 0.0
                    )
                    if use_strict_pfs:
                        phi_rnd = base_phase + np.random.uniform(-np.pi, np.pi, size=num_effective_probes)
                    else:
                        phi_rnd = base_phase + np.random.uniform(-np.pi, np.pi)

                    # Power per ray
                    p_ray = cluster.power / 20.0

                    # Base Weight (without random phase, for later reconstruction).
                    # W_base[p_eff, t] = pol_factor[p_eff] * sqrt(P) * w_map[p_eff] * g_tx[t]
                    # In dual-pol: V probe ports (even) get pol_factor=m_col[0],
                    # H probe ports (odd) get m_col[1]. Single-pol: pol_factor=1.
                    W_base = pol_factor_col * np.sqrt(p_ray) * (w_map_ray @ g_tx_ray)

                    # Apply random phase for current snapshot.
                    # ray:        W_base (P, Tx) * scalar      -> (P, Tx)
                    # strict_pfs: W_base (P, Tx) * (P, 1)       -> (P, Tx) per-probe independent
                    if use_strict_pfs:
                        A_ray = W_base * np.exp(1j * phi_rnd)[:, None]
                    else:
                        A_ray = W_base * np.exp(1j * phi_rnd)
                    
                    A_total += A_ray
                    cluster_A_accum += A_ray # Accumulate for tap
                    cluster_power_accum += (w_map_ray.flatten() * p_ray)
                    
                    # Calculate Doppler for this ray
                    # fd = v/lambda * cos(AoA - theta_v)
                    # UE velocity vector is (vx, vy, vz). 
                    # We need angle of arrival vector k_hat_rx.
                    # k_hat_rx points INTO the UE. 
                    # Doppler shift = (v . k_hat_rx) / lambda.
                    # k_hat_rx (spherical) from AoA/ZoA.
                    theta_r_rad = np.deg2rad(zoa_ray) # ZoA (from Zenith)
                    phi_r_rad = np.deg2rad(aoa_ray) # AoA
                    
                    k_hat_rx = np.array([
                        np.sin(theta_r_rad) * np.cos(phi_r_rad),
                        np.sin(theta_r_rad) * np.sin(phi_r_rad),
                        np.cos(theta_r_rad)
                    ])
                    
                    # UE Velocity
                    v_vec = channel_profile.ue_velocity # tuple or array
                    v_dot_k = np.dot(v_vec, k_hat_rx)
                    fd = v_dot_k / lambda_0
                    
                    ray_components.append({
                        'W_base': W_base, # Complex (N_probes, N_tx)
                        'phi_init': phi_rnd,
                        'doppler_hz': fd,
                        'cluster_index': i, # Link to parent cluster/tap
                        # phi_axis tells downstream consumers (e.g. exporter) which
                        # axis of W_base the per-probe phi vector indexes. For DL,
                        # W_base is (N_probes, N_tx) so phi (when vector) aligns
                        # with rows. The UL mirror in _build_ul_ray_components
                        # transposes W and flips this to 'col'. Scalar phi (ray
                        # mode) ignores this field.
                        'phi_axis': 'row',
                    })

                cluster_probe_mapping[i] = cluster_power_accum
                
                # Store Wideband Tap (Aggregated Rays)
                impulse_response.append({
                    'delay': cluster.delay,
                    'coeff': cluster_A_accum,
                    'cluster_index': i
                })

            transmit_mapping_matrix_A = A_total
            probe_tx_power_contributions = np.abs(transmit_mapping_matrix_A)**2
            
        else:
            raise ValueError(f"Unknown synthesis method: {synthesis_method}")

        # Apply Probe Phase Jitter (Simulating hardware inconsistency)
        if probe_phase_jitter:
            if synthesis_method == 'strict_pfs':
                warnings.warn(
                    "probe_phase_jitter=True with synthesis_method='strict_pfs': "
                    "strict_pfs already produces per-(probe, cluster) independent fading "
                    "(per-realization strict PFS); adding jitter is redundant and pollutes the "
                    "intentional phase structure. Set probe_phase_jitter=False with strict_pfs.",
                    stacklevel=2,
                )
            if chamber_config.calibration_vector is not None:
                warnings.warn(
                    "probe_phase_jitter=True with calibration_vector present: "
                    "jitter applies uniform [-pi, pi] random phase per probe, which will "
                    "swamp the per-probe static phase correction provided by calibration_vector. "
                    "These two mechanisms are effectively mutually exclusive; the calibration "
                    "phase information is silently lost. Disable one of them.",
                    stacklevel=2,
                )
            # Jitter range: Full random phase [-pi, pi]
            # This simulates uncalibrated phase or large cable length variations,
            # which helps break the strong coherence of the simplified Cluster model.
            jitter_rad = np.pi
            phase_noise = np.random.uniform(-jitter_rad, jitter_rad, size=(num_effective_probes, 1))
            transmit_mapping_matrix_A *= np.exp(1j * phase_noise)
            # Note: Jitter is applied to the TOTAL snapshot. 
            # If we reconstruct from rays, we must apply jitter there too.
            # We should store the jitter vector to apply it consistently.
            # Let's add jitter vector to return dict.
            jitter_vector = np.exp(1j * phase_noise)
        else:
            jitter_vector = None

        effective_H_matrix = B_matrix @ transmit_mapping_matrix_A

        target_H_matrix_ul = None
        transmit_mapping_matrix_A_ul = None
        effective_H_matrix_ul = None
        channel_impulse_response_ul = None
        ray_components_ul = None
        if channel_config.enable_ul_dl:
            if target_H_matrix is not None:
                target_H_matrix_ul = self._apply_reciprocity_transform(
                    target_H_matrix, channel_config.reciprocity_transform
                )
            if transmit_mapping_matrix_A is not None:
                transmit_mapping_matrix_A_ul = self._apply_reciprocity_transform(
                    transmit_mapping_matrix_A, channel_config.reciprocity_transform
                )
                effective_H_matrix_ul = transmit_mapping_matrix_A_ul @ B_matrix.T
            if impulse_response:
                channel_impulse_response_ul = self._build_ul_impulse_response(
                    impulse_response, channel_config.reciprocity_transform
                )
            ray_components_ul = self._build_ul_ray_components(
                ray_components, channel_config.reciprocity_transform
            )
        return {
            "center_frequency_hz": channel_profile.center_frequency_hz,
            "target_H_matrix": target_H_matrix,
            "transmit_mapping_matrix_A": transmit_mapping_matrix_A,
            "effective_H_matrix": effective_H_matrix,
            "B_matrix": B_matrix,
            "clusters": physical_clusters,
            "lsps": lsps, # Return LSPs for display
            "pathloss_db": pathloss_db,
            "is_los": is_los,
            "probe_tx_power_contributions": probe_tx_power_contributions,
            "cluster_probe_mapping": cluster_probe_mapping,
            "ray_components": ray_components, # Explicit variable
            "probe_phase_jitter_vector": jitter_vector,
            "channel_impulse_response": impulse_response, # Wideband Taps
            "target_H_matrix_ul": target_H_matrix_ul,
            "transmit_mapping_matrix_A_ul": transmit_mapping_matrix_A_ul,
            "effective_H_matrix_ul": effective_H_matrix_ul,
            "channel_impulse_response_ul": channel_impulse_response_ul,
            "ray_components_ul": ray_components_ul,
            "reciprocity_transform": channel_config.reciprocity_transform,
            "ul_enabled": channel_config.enable_ul_dl,
        }
