"""
Cable (Channel) Simulation Class
This is the infrastructure provided by the course to simulate physical transmission medium
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


class Cable:
    """
    Cable/Channel Simulation Class

    Features:
    1. Transmit analog signals
    2. Add channel noise
    3. Simulate signal attenuation
    4. Debug mode: Visualize signal waveforms
    """

    def __init__(
        self,
        length: float = 100.0,
        attenuation: float = 0.1,
        noise_level: float = 0.01,
        debug_mode: bool = False,
    ):
        """
        Initialize the cable

        Args:
            length: Cable length (meters)
            attenuation: Attenuation coefficient (dB/m)
            noise_level: Noise level (for adding Gaussian white noise)
            debug_mode: Debug mode, will display signal waveforms when enabled
        """
        self.length = length
        self.attenuation = attenuation
        self.noise_level = noise_level
        self.debug_mode = debug_mode

        # Store the most recent transmitted signals (for debugging)
        self.last_input_signal: Optional[np.ndarray] = None
        self.last_output_signal: Optional[np.ndarray] = None

    def transmit(self, signal: np.ndarray) -> np.ndarray:
        """
        Transmit signal through the cable

        Args:
            signal: Input analog signal (numpy array)

        Returns:
            Signal after passing through the channel (with attenuation and noise)
        """
        # Save input signal for debugging
        self.last_input_signal = signal.copy()

        # 1. Apply attenuation
        # Using exponential attenuation model: A(d) = A0 * exp(-α * d)
        attenuation_factor = np.exp(-self.attenuation * self.length / 100)
        attenuated_signal = signal * attenuation_factor

        # 2. Add Gaussian white noise
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level, len(signal))
            noisy_signal = attenuated_signal + noise
        else:
            noisy_signal = attenuated_signal

        # Save output signal for debugging
        self.last_output_signal = noisy_signal.copy()

        # Display waveforms if debug mode is enabled
        if self.debug_mode:
            self.plot_signals()

        return noisy_signal

    def get_propagation_delay(self, signal_speed: float = 2e8) -> float:
        """
        Calculate propagation delay

        Args:
            signal_speed: Signal propagation speed (m/s)
                         Default is 2/3 of light speed (typical for fiber optics)

        Returns:
            Propagation delay (seconds)
        """
        return self.length / signal_speed

    def plot_signals(self, max_samples: int = 1000):
        """
        Plot signal waveforms (debug feature)

        Args:
            max_samples: Maximum number of samples to display (to avoid oversized plots)
        """
        if self.last_input_signal is None or self.last_output_signal is None:
            print("No signal data available")
            return

        # Limit the number of samples displayed
        input_signal = self.last_input_signal[:max_samples]
        output_signal = self.last_output_signal[:max_samples]

        # Create figure
        plt.figure(figsize=(12, 6))

        # Subplot 1: Input signal
        plt.subplot(2, 1, 1)
        plt.plot(input_signal, "b-", linewidth=0.5)
        plt.title("Input Signal (Transmitter)")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)

        # Subplot 2: Output signal
        plt.subplot(2, 1, 2)
        plt.plot(output_signal, "r-", linewidth=0.5)
        plt.title(
            f"Output Signal (Receiver) - Attenuation={self.attenuation}, Noise={self.noise_level}"
        )
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def get_signal_stats(self) -> dict:
        """
        Get signal statistics (debug feature)

        Returns:
            Dictionary containing signal statistics
        """
        if self.last_input_signal is None or self.last_output_signal is None:
            return {}

        stats = {
            "input_mean": np.mean(self.last_input_signal),
            "input_std": np.std(self.last_input_signal),
            "input_max": np.max(self.last_input_signal),
            "input_min": np.min(self.last_input_signal),
            "output_mean": np.mean(self.last_output_signal),
            "output_std": np.std(self.last_output_signal),
            "output_max": np.max(self.last_output_signal),
            "output_min": np.min(self.last_output_signal),
            "snr_db": self._calculate_snr(),
        }

        return stats

    def _calculate_snr(self) -> float:
        """
        Calculate Signal-to-Noise Ratio (SNR)

        Returns:
            SNR in dB
        """
        if self.last_input_signal is None or self.last_output_signal is None:
            return 0.0

        # Noise = Output - Input * attenuation_factor
        attenuation_factor = np.exp(-self.attenuation * self.length / 100)
        expected_output = self.last_input_signal * attenuation_factor
        noise = self.last_output_signal - expected_output

        signal_power = np.mean(expected_output**2)
        noise_power = np.mean(noise**2)

        if noise_power == 0:
            return float("inf")

        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    def __str__(self) -> str:
        return (
            f"Cable(length={self.length}m, "
            f"attenuation={self.attenuation}dB/m, "
            f"noise_level={self.noise_level}, "
            f"debug={self.debug_mode})"
        )


# ============================================================================
# Usage Example
# ============================================================================


def example_usage():
    """Example: How to use the Cable class"""

    print("=" * 60)
    print("Cable Class Usage Example")
    print("=" * 60)

    # Create cable (with debug mode enabled)
    cable = Cable(
        length=100,  # 100 meters
        attenuation=0.1,  # Attenuation coefficient
        noise_level=0.1,  # Noise level
        debug_mode=True,  # Set to True to see waveforms
    )

    print(f"\n{cable}")

    # Generate a simple sine wave signal as an example
    sample_rate = 44100  # Sampling rate
    duration = 0.5  # Duration (seconds)
    frequency = 1000  # Frequency (Hz)

    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = np.sin(2 * np.pi * frequency * t)

    print(f"\nTransmitting signal:")
    print(f"  Number of samples: {len(signal)}")
    print(f"  Frequency: {frequency} Hz")

    # Transmit through cable
    received_signal = cable.transmit(signal)

    print(f"\nReceived signal:")
    print(f"  Number of samples: {len(received_signal)}")
    print(f"  Propagation delay: {cable.get_propagation_delay():.9f} seconds")

    # Get statistics
    stats = cable.get_signal_stats()
    print(f"\nSignal statistics:")
    print(f"  Input signal mean: {stats['input_mean']:.6f}")
    print(f"  Output signal mean: {stats['output_mean']:.6f}")
    print(f"  SNR: {stats['snr_db']:.2f} dB")

    # Uncomment below to view waveforms
    # cable.plot_signals()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    example_usage()
