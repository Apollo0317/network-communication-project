"""
Unit tests for Modulator and DeModulator classes
Tests basic functionality and calculates BER/SER under different channel conditions
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt
from modulator import Modulator, DeModulator
from cable import Cable
import os
from typing import Tuple, List


class TestModulatorBasic(unittest.TestCase):
    """Basic functionality tests"""

    def setUp(self):
        """Set up test fixtures"""
        self.modulator = Modulator(
            scheme="16QAM", symbol_rate=1e6, sample_rate=50e6, fc=2e6
        )
        self.demodulator = DeModulator(
            scheme="16QAM", symbol_rate=1e6, sample_rate=50e6, fc=2e6
        )

    def test_bytes_to_bits(self):
        """Test byte to bit conversion"""
        test_data = b"\x00\xff\xaa"
        expected = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,  # 0x00
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,  # 0xFF
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
        ]  # 0xAA
        result = Modulator.bytes_to_bits(test_data)
        self.assertEqual(result, expected)

    def test_bits_to_bytes(self):
        """Test bit to byte conversion"""
        test_bits = [
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,  # 0x55
            1,
            1,
            0,
            0,
            0,
            0,
            1,
            1,
        ]  # 0xC3
        expected = b"\x55\xc3"
        result = DeModulator.bits_to_bytes(test_bits)
        self.assertEqual(result, expected)

    def test_qam_mapping_size(self):
        """Test QAM mapping dictionary size"""
        mapping = Modulator.generate_QAM_mapping()
        self.assertEqual(len(mapping), 16)

    def test_qam_mapping_symmetry(self):
        """Test QAM constellation symmetry"""
        mapping = Modulator.generate_QAM_mapping()
        values = list(mapping.values())
        I_values = [v[0] for v in values]
        Q_values = [v[1] for v in values]

        # Check if constellation points are symmetric
        self.assertEqual(set(I_values), {-3, -1, 1, 3})
        self.assertEqual(set(Q_values), {-3, -1, 1, 3})

    def test_modulate_output_type(self):
        """Test modulate output type"""
        test_data = b"Test"
        signal = self.modulator.modulate(test_data)
        self.assertIsInstance(signal, np.ndarray)
        self.assertTrue(len(signal) > 0)

    def test_perfect_channel(self):
        """Test with perfect channel (no noise, no attenuation)"""
        test_data = b"Hello World!"

        # Modulate
        qam_signal = self.modulator.modulate(test_data)

        # Perfect channel
        cable = Cable(length=0, attenuation=0, noise_level=0, debug_mode=False)
        recv_signal = cable.transmit(qam_signal)

        # Demodulate
        recv_data = self.demodulator.demodulate(recv_signal)

        # Should match exactly (with possible zero padding)
        self.assertTrue(recv_data.startswith(test_data))

    def test_different_data_lengths(self):
        """Test with different data lengths"""
        test_cases = [
            b"A",  # 1 byte
            b"Test",  # 4 bytes
            b"Hello World!",  # 12 bytes
            b"X" * 100,  # 100 bytes
        ]

        cable = Cable(length=10, attenuation=0.1, noise_level=0.01, debug_mode=False)

        for test_data in test_cases:
            with self.subTest(length=len(test_data)):
                qam_signal = self.modulator.modulate(test_data)
                recv_signal = cable.transmit(qam_signal)
                recv_data = self.demodulator.demodulate(recv_signal)

                # Check length (accounting for zero padding)
                self.assertGreaterEqual(len(recv_data), len(test_data))


class TestErrorRateAnalysis(unittest.TestCase):
    """Error rate analysis under different channel conditions"""

    def setUp(self):
        """Set up test fixtures"""
        self.modulator = Modulator(
            scheme="16QAM", symbol_rate=1e6, sample_rate=50e6, fc=2e6
        )
        self.demodulator = DeModulator(
            scheme="16QAM", symbol_rate=1e6, sample_rate=50e6, fc=2e6
        )
        # Use a longer test string for better statistics
        self.test_data = b"The quick brown fox jumps over the lazy dog. " * 10

    def calculate_ber_ser(
        self, sent_data: bytes, recv_data: bytes
    ) -> Tuple[float, float]:
        """
        Calculate Bit Error Rate (BER) and Symbol Error Rate (SER)

        Args:
            sent_data: Original sent data
            recv_data: Received data

        Returns:
            Tuple of (BER, SER)
        """
        # Convert to bits
        sent_bits = Modulator.bytes_to_bits(sent_data)
        recv_bits = Modulator.bytes_to_bits(recv_data[: len(sent_data)])  # Trim padding

        # Calculate BER
        min_len = min(len(sent_bits), len(recv_bits))
        bit_errors = sum(
            s != r for s, r in zip(sent_bits[:min_len], recv_bits[:min_len])
        )
        ber = bit_errors / min_len if min_len > 0 else 0

        # Calculate SER (4 bits per symbol for 16QAM)
        symbol_errors = 0
        total_symbols = min_len // 4
        for i in range(total_symbols):
            sent_symbol = sent_bits[i * 4 : (i + 1) * 4]
            recv_symbol = recv_bits[i * 4 : (i + 1) * 4]
            if sent_symbol != recv_symbol:
                symbol_errors += 1

        ser = symbol_errors / total_symbols if total_symbols > 0 else 0

        return ber, ser

    def test_noise_levels(self):
        """Test different noise levels"""
        noise_levels = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]
        results = []

        print("\n" + "=" * 70)
        print("Testing Different Noise Levels (Length=100m, Attenuation=0.1)")
        print("=" * 70)
        print(f"{'Noise Level':<15} {'BER':<15} {'SER':<15} {'Status':<20}")
        print("-" * 70)

        for noise in noise_levels:
            cable = Cable(
                length=100, attenuation=0.1, noise_level=noise, debug_mode=False
            )

            qam_signal = self.modulator.modulate(self.test_data)
            recv_signal = cable.transmit(qam_signal)
            recv_data = self.demodulator.demodulate(recv_signal)

            ber, ser = self.calculate_ber_ser(self.test_data, recv_data)
            results.append((noise, ber, ser))

            status = "✓ Good" if ber < 0.01 else "✗ Poor" if ber < 0.1 else "✗ Failed"
            print(f"{noise:<15.2f} {ber:<15.6f} {ser:<15.6f} {status:<20}")

        print("=" * 70)

        # Store results for plotting
        self.noise_test_results = results

    def test_attenuation_levels(self):
        """Test different attenuation levels"""
        attenuation_levels = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
        results = []

        print("\n" + "=" * 70)
        print("Testing Different Attenuation Levels (Length=100m, Noise=0.1)")
        print("=" * 70)
        print(f"{'Attenuation':<15} {'BER':<15} {'SER':<15} {'Status':<20}")
        print("-" * 70)

        for atten in attenuation_levels:
            cable = Cable(
                length=100, attenuation=atten, noise_level=0.1, debug_mode=False
            )

            qam_signal = self.modulator.modulate(self.test_data)
            recv_signal = cable.transmit(qam_signal)
            recv_data = self.demodulator.demodulate(recv_signal)

            ber, ser = self.calculate_ber_ser(self.test_data, recv_data)
            results.append((atten, ber, ser))

            status = "✓ Good" if ber < 0.01 else "✗ Poor" if ber < 0.1 else "✗ Failed"
            print(f"{atten:<15.2f} {ber:<15.6f} {ser:<15.6f} {status:<20}")

        print("=" * 70)

        # Store results for plotting
        self.atten_test_results = results

    def test_cable_lengths(self):
        """Test different cable lengths"""
        lengths = [10, 50, 100, 200, 500, 1000]
        results = []

        print("\n" + "=" * 70)
        print("Testing Different Cable Lengths (Attenuation=0.1, Noise=0.1)")
        print("=" * 70)
        print(f"{'Length (m)':<15} {'BER':<15} {'SER':<15} {'Status':<20}")
        print("-" * 70)

        for length in lengths:
            cable = Cable(
                length=length, attenuation=0.1, noise_level=0.1, debug_mode=False
            )

            qam_signal = self.modulator.modulate(self.test_data)
            recv_signal = cable.transmit(qam_signal)
            recv_data = self.demodulator.demodulate(recv_signal)

            ber, ser = self.calculate_ber_ser(self.test_data, recv_data)
            results.append((length, ber, ser))

            status = "✓ Good" if ber < 0.01 else "✗ Poor" if ber < 0.1 else "✗ Failed"
            print(f"{length:<15} {ber:<15.6f} {ser:<15.6f} {status:<20}")

        print("=" * 70)

        # Store results for plotting
        self.length_test_results = results

    def test_combined_conditions(self):
        """Test various combined channel conditions"""
        test_cases = [
            # (length, attenuation, noise, description)
            (50, 0.05, 0.05, "Ideal conditions"),
            (100, 0.1, 0.1, "Normal conditions"),
            (200, 0.25, 0.3, "Moderate degradation"),
            (500, 0.5, 0.5, "Poor conditions"),
            (1000, 1.0, 1.0, "Extreme conditions"),
        ]

        print("\n" + "=" * 90)
        print("Testing Combined Channel Conditions")
        print("=" * 90)
        print(
            f"{'Description':<25} {'Len(m)':<8} {'Atten':<8} {'Noise':<8} {'BER':<12} {'SER':<12}"
        )
        print("-" * 90)

        for length, atten, noise, desc in test_cases:
            cable = Cable(
                length=length, attenuation=atten, noise_level=noise, debug_mode=False
            )

            qam_signal = self.modulator.modulate(self.test_data)
            recv_signal = cable.transmit(qam_signal)
            recv_data = self.demodulator.demodulate(recv_signal)

            ber, ser = self.calculate_ber_ser(self.test_data, recv_data)

            print(
                f"{desc:<25} {length:<8} {atten:<8.2f} {noise:<8.2f} {ber:<12.6f} {ser:<12.6f}"
            )

        print("=" * 90)


class TestErrorRatePlots(unittest.TestCase):
    """Generate error rate plots"""

    def setUp(self):
        """Set up test fixtures"""
        self.modulator = Modulator(
            scheme="16QAM", symbol_rate=1e6, sample_rate=50e6, fc=2e6
        )
        self.demodulator = DeModulator(
            scheme="16QAM", symbol_rate=1e6, sample_rate=50e6, fc=2e6
        )
        self.test_data = b"Testing error rates with 16QAM modulation scheme. " * 20

        # Create output directory for plots
        self.output_dir = "test_results"
        os.makedirs(self.output_dir, exist_ok=True)

    def calculate_ber_ser(
        self, sent_data: bytes, recv_data: bytes
    ) -> Tuple[float, float]:
        """Calculate BER and SER"""
        sent_bits = Modulator.bytes_to_bits(sent_data)
        recv_bits = Modulator.bytes_to_bits(recv_data[: len(sent_data)])

        min_len = min(len(sent_bits), len(recv_bits))
        bit_errors = sum(
            s != r for s, r in zip(sent_bits[:min_len], recv_bits[:min_len])
        )
        ber = bit_errors / min_len if min_len > 0 else 0

        symbol_errors = 0
        total_symbols = min_len // 4
        for i in range(total_symbols):
            sent_symbol = sent_bits[i * 4 : (i + 1) * 4]
            recv_symbol = recv_bits[i * 4 : (i + 1) * 4]
            if sent_symbol != recv_symbol:
                symbol_errors += 1

        ser = symbol_errors / total_symbols if total_symbols > 0 else 0

        return ber, ser

    def test_generate_snr_vs_ber_plot(self):
        """Generate SNR vs BER/SER plot"""
        snr_db_range = np.arange(0, 30, 2)
        ber_list = []
        ser_list = []

        print("\n" + "=" * 60)
        print("Generating SNR vs BER/SER curve...")
        print("=" * 60)

        for snr_db in snr_db_range:
            # Convert SNR to noise level (approximate)
            noise_level = 10 ** (-snr_db / 20) * 0.5

            cable = Cable(
                length=100, attenuation=0.1, noise_level=noise_level, debug_mode=False
            )

            qam_signal = self.modulator.modulate(self.test_data)
            recv_signal = cable.transmit(qam_signal)
            recv_data = self.demodulator.demodulate(recv_signal)

            ber, ser = self.calculate_ber_ser(self.test_data, recv_data)
            ber_list.append(ber)
            ser_list.append(ser)

            print(f"SNR: {snr_db:5.1f} dB, BER: {ber:.6f}, SER: {ser:.6f}")

        # Plot
        plt.figure(figsize=(10, 6))
        plt.semilogy(snr_db_range, ber_list, "b-o", label="BER", markersize=4)
        plt.semilogy(snr_db_range, ser_list, "r-s", label="SER", markersize=4)
        plt.xlabel("SNR (dB)")
        plt.ylabel("Error Rate")
        plt.title("16QAM: SNR vs BER/SER")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, "snr_vs_ber_ser.png")
        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"Plot saved to: {output_path}")
        print("=" * 60)

    def test_generate_comparison_plot(self):
        """Generate comprehensive comparison plot"""
        # Test different parameters
        noise_levels = [0.0, 0.1, 0.3, 0.5, 1.0]
        noise_results = []

        for noise in noise_levels:
            cable = Cable(
                length=100, attenuation=0.1, noise_level=noise, debug_mode=False
            )
            qam_signal = self.modulator.modulate(self.test_data)
            recv_signal = cable.transmit(qam_signal)
            recv_data = self.demodulator.demodulate(recv_signal)
            ber, ser = self.calculate_ber_ser(self.test_data, recv_data)
            noise_results.append((noise, ber, ser))

        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # BER plot
        noise_vals = [r[0] for r in noise_results]
        ber_vals = [r[1] for r in noise_results]
        ser_vals = [r[2] for r in noise_results]

        axes[0].semilogy(noise_vals, ber_vals, "b-o", label="BER", linewidth=2)
        axes[0].set_xlabel("Noise Level")
        axes[0].set_ylabel("Bit Error Rate")
        axes[0].set_title("Noise Level vs BER")
        axes[0].grid(True, which="both", alpha=0.3)
        axes[0].legend()

        axes[1].semilogy(noise_vals, ser_vals, "r-s", label="SER", linewidth=2)
        axes[1].set_xlabel("Noise Level")
        axes[1].set_ylabel("Symbol Error Rate")
        axes[1].set_title("Noise Level vs SER")
        axes[1].grid(True, which="both", alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "error_rate_comparison.png")
        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"\nComparison plot saved to: {output_path}")


def run_all_tests():
    """Run all test suites"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestModulatorBasic))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorRateAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorRatePlots))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("16QAM Modulator/Demodulator Unit Tests")
    print("=" * 70)

    result = run_all_tests()

    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)
