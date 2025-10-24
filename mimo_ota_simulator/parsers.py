
import configparser
from pathlib import Path
from typing import List
from mimo_ota_simulator.data_models import ChannelProfile, Cluster

class CDL_Parser:
    """
    Parses a CDL (Channel Delay Line) model definition file.
    """
    def parse(self, file_path: Path) -> ChannelProfile:
        """
        Parses a .ini file representing a CDL channel model.

        Args:
            file_path: The path to the CDL .ini file.

        Returns:
            A ChannelProfile object containing the parsed channel information.
        """
        config = configparser.ConfigParser()
        config.read(file_path)

        clusters: List[Cluster] = []
        for section in config.sections():
            if section.startswith('cluster-'):
                try:
                    cluster = Cluster(
                        aod_deg=config.getfloat(section, 'aod_deg'),
                        aoa_deg=config.getfloat(section, 'aoa_deg'),
                        power_dB=config.getfloat(section, 'power_dB'),
                        delay_s=config.getfloat(section, 'delay_s'),
                    )
                    clusters.append(cluster)
                except (configparser.NoOptionError, ValueError) as e:
                    raise ValueError(f"Error parsing section {section}: {e}") from e
        
        if not clusters:
            raise ValueError(f"No clusters found in the file: {file_path}")

        return ChannelProfile(clusters=clusters)
