"""
Physical Layer Simulation Entities
"""

from collections import deque
from typing import Optional, List
import numpy as np
from core import SimulationEntity, PhySimulationEngine
from phy.cable import Cable
from phy.modulator import Modulator, DeModulator


class ChannelEntity(SimulationEntity):
    """Transmission media, connects TxEntity and RxEntity"""

    def __init__(self, cable:Cable, name: str = "Channel"):
        super().__init__(name)
        self.cable = cable

        propagation_delay_s = cable.get_propagation_delay()
        time_step_us = 1.0
        propagation_delay_ticks = int(propagation_delay_s / (time_step_us * 1e-6))
        self.propagation_delay_ticks = propagation_delay_ticks

        self.receivers: List["RxEntity"] = []

        # signal queue: (signal, arrival_tick)
        self.transit_queue: deque = deque()

        self.stats = {"signals_in_transit": 0, "signals_delivered": 0}

    def connect_receiver(self, rx: "RxEntity"):
        self.receivers.append(rx)

    def accept_signal(self, signal: np.ndarray, current_tick: int):
        """供 Tx 调用：接收信号进入信道"""

        # 计算信号到达接收端的时间（传播延迟）
        arrival_tick = current_tick + self.propagation_delay_ticks
        self.transit_queue.append((signal, arrival_tick))
        self.stats["signals_in_transit"] += 1

    def update(self, tick: int):
        super().update(tick)

        # 检查是否有信号到达接收端
        while self.transit_queue and self.transit_queue[0][1] <= tick:
            signal, _ = self.transit_queue.popleft()

            # 1. 应用信道损伤 (Cable 模型)
            damaged_signal = self.cable.transmit(signal)

            # 2. 推送给所有连接的接收端
            for rx in self.receivers:
                rx.on_signal_arrival(damaged_signal)

            self.stats["signals_delivered"] += 1
            self.stats["signals_in_transit"] -= 1

    def get_stats(self):
        return self.stats


class TxEntity(SimulationEntity):
    """Transmitter entity"""

    def __init__(self, modulator:Modulator, name: str = "Transmitter"):
        super().__init__(name)
        self.modulator = modulator
        self.tx_buffer = deque()
        self.connected_channel: Optional[ChannelEntity] = None

        self.is_transmitting = False
        self.transmission_end_tick = 0

        self.stats = {"bytes_sent": 0, "packets_sent": 0}

    def connect_to_channel(self, channel: ChannelEntity):
        self.connected_channel = channel

    def enqueue_data(self, data: bytes):
        self.tx_buffer.append(data)

    def update(self, tick: int):
        super().update(tick)

        # 1. 检查是否正在传输中 (模拟传输时延)
        if self.is_transmitting:
            if tick >= self.transmission_end_tick:
                self.is_transmitting = False
            else:
                return  # 正在忙，不能发送下一个包

        # 2. 如果空闲且有数据，开始发送
        if self.tx_buffer and self.connected_channel:
            data = self.tx_buffer.popleft()

            # 调制
            signal = self.modulator.modulate(data)

            # 计算传输时延 (Transmission Delay) = 样本数 / 采样率
            # 假设 time_step_us = 1.0 (需要在外部保证或传入)
            # 这里简化处理，假设 tick = 1us
            # 信号长度 / 50MHz = 秒数 -> 换算成 us (ticks)
            duration_sec = len(signal) / self.modulator.sample_rate
            duration_ticks = int(duration_sec * 1e6)

            # 设置忙碌状态
            self.is_transmitting = True
            self.transmission_end_tick = tick + duration_ticks

            # 将信号推入信道
            self.connected_channel.accept_signal(signal, tick)

            self.stats["bytes_sent"] += len(data)
            self.stats["packets_sent"] += 1

    def get_stats(self):
        return self.stats


class RxEntity(SimulationEntity):
    """Receiver entity"""

    def __init__(self, demodulator:DeModulator, name: str = "Receiver"):
        super().__init__(name)
        self.demodulator = demodulator
        self.rx_buffer = deque()  # 存放解调后的字节流
        self.stats = {"bytes_received": 0}

    def on_signal_arrival(self, signal: np.ndarray):
        """回调函数：当信道有信号送达时被调用"""
        # 立即解调 (或者放入缓冲区等待 update 处理，取决于是否模拟接收处理延迟)
        # 这里简化为立即解调
        data = self.demodulator.demodulate(signal)
        self.rx_buffer.append(data)
        self.stats["bytes_received"] += len(data)

    def get_received_data(self) -> Optional[bytes]:
        if self.rx_buffer:
            return self.rx_buffer.popleft()
        return None

    def get_stats(self):
        return self.stats


class TwistedPair:
    """
    encapsulates a twisted pair cable with two channels: A and B

    allows two devices to connect with automatic crossover handling
    """

    def __init__(self, cable: Cable, simulator: PhySimulationEngine, ID:int=0):
        channel_a = ChannelEntity(cable=cable, name="channel_a")
        channel_b = ChannelEntity(cable=cable, name="channel_b")
        simulator.register_entity(entity=channel_a)
        simulator.register_entity(entity=channel_b)
        self.channel_a = channel_a
        self.channel_b = channel_b
        self.connected_count = 0
        self.ID= ID

    def __str__(self):
        return f"TwistedPair_{self.ID}(Channel A, Channel B)"

    def connect(self, tx_interface: TxEntity, rx_interface: RxEntity):
        """
        将物理层接口连接到双绞线。
        自动处理交叉连接：
        - 第1个设备:Tx -> Channel A, Channel B -> Rx
        - 第2个设备:Tx -> Channel B, Channel A -> Rx
        """
        if self.connected_count == 0:
            tx_interface.connect_to_channel(self.channel_a)
            self.channel_b.connect_receiver(rx_interface)
            self.connected_count += 1
        elif self.connected_count == 1:
            tx_interface.connect_to_channel(self.channel_b)
            self.channel_a.connect_receiver(rx_interface)
            self.connected_count += 1
        else:
            raise ConnectionError(
                "TwistedPair is already fully connected (max 2 devices)."
            )
