"""
Modulator and DeModulator for 16-QAM scheme

Provides phy layer interface
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import signal
import json
import time
import numba
from phy.cable import Cable


class Modulator:
    """
    16-QAM Modulator
    
    Modulates byte data into QAM signals for physical layer transmission
    """
    
    def __init__(
        self,
        scheme: str,
        symbol_rate: int,
        sample_rate: int,
        fc: int,
        power_factor: int = 100,
    ):
        """
        Initialize the modulator
        
        Args:
            scheme: Modulation scheme (e.g., "16QAM")
            symbol_rate: Symbol rate in symbols/second
            sample_rate: Sampling rate in samples/second
            fc: Carrier frequency in Hz
            power_factor: Power amplification factor, default 100
        """
        self.scheme = scheme
        self.symbol_rate = symbol_rate
        self.sample_rate = sample_rate
        self.fc = fc
        self.mapping = Modulator.generate_QAM_mapping()
        self.has_estimated = False  # Flag for training symbol transmission
        self.power_factor = power_factor
        pass

    @staticmethod
    def bytes_to_bits(byte_data: bytes) -> list:
        """
        Convert byte data to bit list
        
        Args:
            byte_data: Input byte data
            
        Returns:
            Bit list, each byte converted to 8 bits (MSB first)
        """
        bit_list = []
        for byte in byte_data:
            # Extract bits from MSB to LSB
            bit_list.extend([(byte >> i) & 1 for i in range(7, -1, -1)])
        return bit_list

    @staticmethod
    def generate_QAM_mapping(order: int = 16) -> dict[str, list]:
        """
        Generate 16-QAM constellation mapping
        
        Uses Gray coding to map 4 bits to I/Q channels (2 bits each)
        Constellation points: I, Q ∈ {-3, -1, 1, 3}
        
        Args:
            order: QAM order, default 16
            
        Returns:
            Mapping dictionary, key is bit list string, value is [I, Q] coordinates
        """
        mapping = {}
        for code in range(order):
            # Convert code to 4-bit representation
            bit_list = [(code >> i) & 1 for i in range(3, -1, -1)]
            # First 2 bits map to I channel, last 2 bits map to Q channel
            I_bit, Q_bit = bit_list[:2], bit_list[2:]
            
            # Gray coding mapping: 00->-3, 01->-1, 11->1, 10->3
            if I_bit == [0, 0]:
                I_out = -3
            elif I_bit == [0, 1]:
                I_out = -1
            elif I_bit == [1, 1]:
                I_out = 1
            else:
                I_out = 3
                
            if Q_bit == [0, 0]:
                Q_out = -3
            elif Q_bit == [0, 1]:
                Q_out = -1
            elif Q_bit == [1, 1]:
                Q_out = 1
            else:
                Q_out = 3
                
            mapping[str(bit_list)] = [I_out, Q_out]
        return mapping

    def Set_Frequency(self, fc: int):
        """
        Set carrier frequency
        
        Args:
            fc: New carrier frequency in Hz
            
        Raises:
            ValueError: Raised when frequency is negative
        """
        if fc < 0:
            raise ValueError("invalid frequency")
        self.fc = fc
        print(f"fc set to {self.fc}Hz now")

    def QAM(self, byte_data: bytes) -> list:
        """
        Map byte data to QAM symbols
        
        Adds 4 training symbols at the beginning for channel estimation on first call
        
        Args:
            byte_data: Input byte data
            
        Returns:
            Symbol list, each symbol is [I, Q] coordinates
        """
        bit_list = Modulator.bytes_to_bits(byte_data)
        bit_per_symbol = 4  # 16-QAM uses 4 bits per symbol
        length = len(bit_list)
        remain_bit = length % bit_per_symbol
        
        # Zero padding to align to 4-bit boundaries
        if remain_bit:
            zero_pad = [0] * (bit_per_symbol - remain_bit)
            print(f"pad zero: {len(zero_pad)}")
            bit_list.extend(zero_pad)
            
        # Group bits into chunks of 4
        grouped_bit_list = []
        for i in range(0, len(bit_list), bit_per_symbol):
            grouped_bit_list.append(
                [bit_list[i], bit_list[i + 1], bit_list[i + 2], bit_list[i + 3]]
            )
            
        # Generate symbols using mapping table
        symbols = [self.mapping.get(str(bit_group)) for bit_group in grouped_bit_list]
        
        # Add training symbols on first transmission
        if not self.has_estimated:
            train_symbols = [[1, 3], [3, 1], [-1, -3], [-3, -1]]
            symbols = train_symbols + symbols
            self.has_estimated = True
            print(
                f"first send: add 4 symbol to estimate alpha\ntotal symbol:{len(symbols)}"
            )
        return symbols

    def QAM_UpConverter(self, symbols: list[list], debug=False) -> np.ndarray:
        """
        QAM upconverter: modulate baseband symbols to carrier frequency
        
        Processing steps:
        1. Convert symbols to complex form
        2. Pulse shaping (currently using rectangular filter)
        3. Quadrature modulation to carrier
        
        Args:
            symbols: Symbol list, each symbol is [I, Q]
            debug: Whether to plot debug waveforms
            
        Returns:
            Modulated time-domain RF signal
        """
        # Convert to complex symbols
        symbols: list[complex] = [complex(symbol[0], symbol[1]) for symbol in symbols]
        complex_symbols = np.array(symbols, dtype=np.complex128)
        
        # Calculate samples per symbol
        sps = self.sample_rate / self.symbol_rate
        I_baseband = np.real(complex_symbols)
        Q_baseband = np.imag(complex_symbols)

        # Pulse shaping: rectangular filter (TODO: add RRC filter)
        I_n = np.repeat(I_baseband, repeats=sps)
        Q_n = np.repeat(Q_baseband, repeats=sps)

        # Generate carrier waves
        lens = len(I_n)
        n = np.arange(lens)
        time = n / self.sample_rate
        time = time * 1e6  # Convert to microseconds
        carrier_cos = np.cos(2 * np.pi * self.fc * n / self.sample_rate)
        carrier_sin = np.sin(2 * np.pi * self.fc * n / self.sample_rate)

        # Quadrature modulation: s(t) = I(t)cos(2πfc*t) - Q(t)sin(2πfc*t)
        I_modulated = I_n * carrier_cos
        Q_modulated = Q_n * (-carrier_sin)
        qam_signal = I_modulated + Q_modulated

        # Plot debug waveforms
        if debug:
            plt.figure(figsize=(12, 6))

            plt.subplot(3, 2, 1)
            plt.plot(time, carrier_cos)
            plt.title("cos carrier")
            plt.xlabel("t/us")
            plt.ylabel("V/volt")
            plt.grid(True)

            plt.subplot(3, 2, 2)
            plt.plot(time, carrier_sin)
            plt.title("sin carrier")
            plt.xlabel("t/us")
            plt.ylabel("V/volt")
            plt.grid(True)

            plt.subplot(3, 2, 3)
            plt.plot(time, I_n)
            plt.title("I(t)")
            plt.xlabel("t/us")
            plt.ylabel("V/volt")
            plt.grid(True)

            plt.subplot(3, 2, 4)
            plt.plot(time, Q_n)
            plt.title("Q(t)")
            plt.xlabel("t/us")
            plt.ylabel("V/volt")
            plt.grid(True)

            plt.subplot(3, 2, 5)
            plt.plot(time, qam_signal)
            plt.title("QAM(t)")
            plt.xlabel("t/us")
            plt.ylabel("V/volt")
            plt.grid(True)

            plt.tight_layout()
            plt.savefig("fig/signal.png")

        return qam_signal

    def modulate(self, data: bytes) -> np.ndarray:
        """
        Modulation entry function
        
        Args:
            data: Byte data to be modulated
            
        Returns:
            Modulated RF signal
        """
        symbols = self.QAM(byte_data=data)
        signal = self.QAM_UpConverter(symbols=symbols, debug=False)
        signal = signal * self.power_factor  # Power amplification
        return signal


class DeModulator:
    """
    16-QAM Demodulator
    
    Demodulates received QAM signals back to byte data
    """
    
    def __init__(
        self,
        scheme: str,
        symbol_rate: int,
        sample_rate: int,
        fc: int,
        power_factor: int = 100,
    ):
        """
        Initialize the demodulator
        
        Args:
            scheme: Modulation scheme
            symbol_rate: Symbol rate
            sample_rate: Sampling rate
            fc: Carrier frequency
            power_factor: Power factor, should match modulator
        """
        self.scheme = scheme
        self.rs = symbol_rate
        self.fs = sample_rate
        self.fc = fc
        self.sps = int(sample_rate / symbol_rate)  # Samples per symbol
        self.mapping: dict[str, list] = DeModulator.generate_QAM_mapping()
        self.aplitude_loss: float = 0  # Channel amplitude loss
        self.has_estimated = False  # Channel estimation flag
        self.power_factor = power_factor

    def generate_QAM_mapping(order: int = 16) -> dict[str, list]:
        """
        Generate reverse QAM mapping (constellation points to bits)
        
        Args:
            order: QAM order
            
        Returns:
            Reverse mapping dictionary, key is [I,Q] coordinate string, value is bit list
        """
        pos_mapping = Modulator.generate_QAM_mapping()
        rev_mapping = {}
        for k, v in pos_mapping.items():
            bit_group: list[int] = json.loads(k)
            rev_mapping[str(v)] = bit_group
        return rev_mapping

    @staticmethod
    def bits_to_bytes(bit_list: list[int]) -> bytes:
        """
        Convert bit list to bytes
        
        Args:
            bit_list: Bit list
            
        Returns:
            Byte data
        """
        bit_num = len(bit_list)
        byte_list = bytearray()
        byte_num = int(bit_num / 8)
        
        for i in range(0, byte_num):
            bits_of_byte = bit_list[i * 8 : (i + 1) * 8]
            value = 0
            # Combine bits to byte (MSB first)
            for j in range(8):
                value += bits_of_byte[j] << (7 - j)
            byte_list.append(value)
        return bytes(byte_list)

    @staticmethod
    @numba.njit
    def distance(a: list[float, 2], b: list[float, 2]) -> float:
        """
        Calculate squared Euclidean distance between two constellation points
        
        Args:
            a: First point [I, Q]
            b: Second point [I, Q]
            
        Returns:
            Squared distance
        """
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

    @staticmethod
    @numba.njit
    def symbol_power(symbol: list[int, 2]) -> float:
        """
        Calculate symbol power
        
        Args:
            symbol: Symbol [I, Q]
            
        Returns:
            Symbol power I²+Q²
        """
        return symbol[0] ** 2 + symbol[1] ** 2

    def Detect_Symbol(self, symbols: list[list]) -> list[int]:
        """
        Symbol detection: map received symbols back to bits
        
        Uses minimum Euclidean distance decision, performs channel estimation on first reception
        
        Args:
            symbols: Received symbol list
            
        Returns:
            Detected bit list
        """
        bit_list = []
        symbol_list: list[str] = self.mapping.keys()
        symbol_list: list[list] = [json.loads(symbol) for symbol in symbol_list]

        # Channel estimation using training symbols on first reception
        if not self.has_estimated:
            std_train_symbols = [[1, 3], [3, 1], [-1, -3], [-3, -1]]
            recv_train_symbols = symbols[:4]
            symbols = symbols[4:]
            
            # Calculate amplitude loss coefficient
            std_symbol_powers = map(self.symbol_power, std_train_symbols)
            recv_symbol_powers = map(self.symbol_power, recv_train_symbols)
            aplitude_loss = math.sqrt(sum(std_symbol_powers) / sum(recv_symbol_powers))
            print(f"aplitude loss:{10 * math.log10(aplitude_loss)} dB")
            self.aplitude_loss = aplitude_loss
            self.has_estimated = True

        # Compensate for channel loss
        fixed_symbols = []
        for symbol in symbols:
            fixed_I, fixed_Q = (
                symbol[0] * self.aplitude_loss,
                symbol[1] * self.aplitude_loss,
            )
            fixed_symbols.append([fixed_I, fixed_Q])

        # Minimum distance decision
        for symbol in fixed_symbols:
            idx = np.argmin(
                [self.distance(symbol, symbol_i) for symbol_i in symbol_list]
            )
            judged_symbol = str(symbol_list[idx])
            bit_list.extend(self.mapping.get(judged_symbol))
        return bit_list

    def QAM_DownConverter(self, qam_signal: np.ndarray, debug=True) -> list[list]:
        """
        QAM downconverter: demodulate RF signal to baseband symbols
        
        Processing steps:
        1. Quadrature demodulation
        2. Low-pass filtering
        3. Symbol sampling
        
        Args:
            qam_signal: Received QAM signal
            debug: Whether to plot debug waveforms
            
        Returns:
            Demodulated symbol list
        """
        lens = len(qam_signal)
        n = np.arange(lens)
        time = n / self.fs
        time_us = time * 1e6
        
        # Generate local carrier
        carrier_cos = np.cos(2 * np.pi * self.fc * n / self.fs)
        carrier_sin = np.sin(2 * np.pi * self.fc * n / self.fs)
        
        # Quadrature demodulation
        I_prime = qam_signal * carrier_cos
        Q_prime = qam_signal * carrier_sin

        # Design low-pass filter
        order = 6
        f_cutoff = 2 * self.rs  # Cutoff frequency is 2x symbol rate
        Wn = f_cutoff / (self.fs / 2)  # Normalized frequency
        b, a = signal.butter(order, Wn, 'low', analog=False)

        # Filter and recover baseband (multiply by 2 to compensate mixing loss)
        I_baseband: list[float] = list(signal.filtfilt(b, a, I_prime) * 2)
        Q_baseband: list[float] = list(signal.filtfilt(b, a, Q_prime) * (-2))

        # Plot I/Q baseband waveforms
        if debug:
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            plt.plot(time_us, I_baseband)
            plt.title("I(t)")
            plt.grid(True)

            plt.subplot(2, 1, 2)
            plt.plot(time_us, Q_baseband)
            plt.title("Q(t)")
            plt.grid(True)

            plt.savefig("fig/IQ.png")

        # Symbol sampling: average over each symbol period
        symbol_num = int(lens / self.sps)
        I_n: list[float] = [
            sum(I_baseband[(i) * self.sps : (i + 1) * self.sps]) / self.sps
            for i in range(symbol_num)
        ]
        Q_n: list[float] = [
            sum(Q_baseband[(i) * self.sps : (i + 1) * self.sps]) / self.sps
            for i in range(symbol_num)
        ]

        symbols = [[I, Q] for (I, Q) in zip(I_n, Q_n)]
        return symbols

    def demodulate(self, signal: np.ndarray) -> bytes:
        """
        Demodulation entry function
        
        Args:
            signal: Received RF signal
            
        Returns:
            Demodulated byte data
        """
        signal = signal / self.power_factor  # Power normalization
        symbols = self.QAM_DownConverter(qam_signal=signal, debug=False)
        bits = self.Detect_Symbol(symbols=symbols)
        byte_recovered = self.bits_to_bytes(bit_list=bits)
        return byte_recovered


def test():
    """
    Test function: verify modulation and demodulation functionality
    
    Now supports both synchronous and asynchronous simulation modes
    """
    import sys
    sys.path.append('..')  # 添加父目录到路径
    
    from core.simulator import PhySimulationEngine, SimulationEntity
    from phy.entity import TxEntity, ChannelEntity, RxEntity
    
    # 创建调制解调器
    modulator = Modulator(
        scheme="16QAM", symbol_rate=1e6, sample_rate=50e6, fc=2e6, power_factor=100
    )
    demodulator = DeModulator(
        scheme="16QAM", symbol_rate=1e6, sample_rate=50e6, fc=2e6, power_factor=100
    )
    
    # 创建信道
    cable = Cable(
        length=100,
        attenuation=2,
        noise_level=3,
        debug_mode=False,
    )
    print(f"\n{cable}")
    
    # 测试数据
    test_str = b"aloha! I like luguan"
    test_sample_num = 1000  # 减少数量以便观察异步行为
    
    # 计算传播延迟（ticks）
    propagation_delay_s = cable.get_propagation_delay()
    time_step_us = 1.0
    propagation_delay_ticks = int(propagation_delay_s / (time_step_us * 1e-6))
    
    print(f"Propagation delay: {propagation_delay_s*1e6:.2f} μs = {propagation_delay_ticks} ticks")
    
    # 创建实体
    tx = TxEntity(modulator, name="Tx-Node")
    channel = ChannelEntity(cable, propagation_delay_ticks, name="Cable-Channel")
    rx = RxEntity(demodulator, name="Rx-Node")
    
    # 预先将数据加入发送队列
    for i in range(test_sample_num):
        tx.enqueue_data(test_str)
    
    # 创建仿真引擎
    engine = PhySimulationEngine(time_step_us=time_step_us, realtime_mode=False)
    
    engine.register_entity(tx)
    engine.register_entity(channel)
    engine.register_entity(rx)

    tx.connect_to_channel(channel)
    channel.connect_receiver(rx)

    # 运行仿真
    # 需要足够的ticks以完成所有传输
    duration_ticks = test_sample_num + propagation_delay_ticks + 100
    
    import time
    start_time = time.time()
    engine.run(duration_ticks=duration_ticks*10)
    cost = time.time() - start_time
    
    # 打印统计
    print("\n" + "="*60)
    print("Simulation Statistics:")
    print(f"Tx: {tx.get_stats()}")
    print(f"Channel: {channel.get_stats()}")
    print(f"Rx: {rx.get_stats()}")
    
    # 验证数据
    success_count = 0
    while rx.rx_buffer:
        recv_data = rx.get_received_data()
        print(f"rx: {recv_data} tx: {test_str}")
        if recv_data == test_str:
            success_count += 1
    
    print(f"\nData verification: {success_count}/{test_sample_num} packets correct")
    print(f"Throughput: {test_sample_num*len(test_str) / 1000 / cost:.2f} KBps")
    print("="*60)


if __name__ == "__main__":
    test()