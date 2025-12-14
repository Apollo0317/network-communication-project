"""
Physical Layer Simulation Entities
"""

from collections import deque
from typing import Optional, List
import numpy as np
from core import SimulationEntity, PhySimulationEngine
from phy.cable import Cable
from phy.modulator import Modulator, DeModulator
from phy.Coding import ChannelEncoder


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
    """Transmitter entity with slicing support"""
    
    PHY_MTU = 256  # 物理层最大传输单元
    
    def __init__(self, modulator: Modulator, name: str = "Transmitter", coding: bool=True):
        super().__init__(name)
        self.modulator = modulator
        self.tx_buffer = deque()
        self.connected_channel: Optional[ChannelEntity] = None
        self.is_transmitting = False
        self.transmission_end_tick = 0
        self.stats = {"bytes_sent": 0, "packets_sent": 0}
        if coding:
            self.encoder= ChannelEncoder()
            self.coding= True
        else:
            self.coding= False
        
        # frame counter
        self._frame_id = 0
    
    def enqueue_data(self, data: bytes):
        slices = self._slice_data(data)
        #print(f"[{self.name}] Enqueued {len(slices)} slices for data of length {len(data)}")
        for s in slices:
            if self.coding:
                s= self.encoder.encoding(s)
            self.tx_buffer.append(s)
    
    def _slice_data(self, data: bytes) -> list[bytes]:
        """
        frame format: [1B frame_id][1B seq][1B total][payload]
        """
        if len(data) <= self.PHY_MTU - 3:
            header = bytes([self._frame_id & 0xFF, 0, 1])
            self._frame_id += 1
            return [header + data]
        
        payload_size = self.PHY_MTU - 3
        total = (len(data) + payload_size - 1) // payload_size
        slices = []
        
        for seq in range(total):
            start = seq * payload_size
            end = min(start + payload_size, len(data))
            # print(f"[{self.name}] Slicing data: slice {seq+1}/{total}, bytes")
            header = bytes([self._frame_id & 0xFF, seq, total])
            slices.append(header + data[start:end])
        
        self._frame_id += 1
        return slices

    def update(self, tick: int):
        super().update(tick)


        if self.is_transmitting:
            if tick >= self.transmission_end_tick:
                self.is_transmitting = False
            else:
                return  

        if self.tx_buffer and self.connected_channel:
            data = self.tx_buffer.popleft()

            signal = self.modulator.modulate(data)

            duration_sec = len(signal) / self.modulator.sample_rate
            duration_ticks = int(duration_sec * 1e6)

            self.is_transmitting = True
            self.transmission_end_tick = tick + duration_ticks

            self.connected_channel.accept_signal(signal, tick)

            self.stats["bytes_sent"] += len(data)
            self.stats["packets_sent"] += 1
    
    def connect_to_channel(self, channel: ChannelEntity):
        self.connected_channel = channel

    def get_stats(self):
        return self.stats


class RxEntity(SimulationEntity):
    """Receiver entity with reassembly support"""
    
    def __init__(self, demodulator: DeModulator, name: str = "Receiver", coding: bool=True):
        super().__init__(name)
        self.demodulator = demodulator
        self.rx_buffer = deque()
        self.stats = {"bytes_received": 0}
        
        # frame_id -> {seq: payload}
        self._reassembly_buffer: dict[int, dict[int, bytes]] = {}

        if coding:
            self.encoder= ChannelEncoder()
            self.coding= True
        else:
            self.coding= False
        # self._expected_total: dict[int, int] = {}
    
    def on_signal_arrival(self, signal: np.ndarray):
        data = self.demodulator.demodulate(signal)

        if self.coding:
            data= self.encoder.decoding(data)

        if not data or len(data) < 3:
            return
        
        frame_id, seq, total = data[0], data[1], data[2]
        payload = data[3:]
        
        if total == 1:
            self.rx_buffer.append(payload)
            self.stats["bytes_received"] += len(payload)
            return
        
        # new buffer for this frame_id
        if frame_id not in self._reassembly_buffer:
            self._reassembly_buffer[frame_id] = {}
        
        self._reassembly_buffer[frame_id][seq] = payload
        print(f"[{self.name}] Received slice {seq+1}/{total} for frame {frame_id}")
        
        # check completion
        if len(self._reassembly_buffer[frame_id]) == total:
            # reassemble
            complete_data = b"".join(
                self._reassembly_buffer[frame_id][i] for i in range(total)
            )
            self.rx_buffer.append(complete_data)
            self.stats["bytes_received"] += len(complete_data)
            
            del self._reassembly_buffer[frame_id]
    
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
        - device 1:Tx -> Channel A, Channel B -> Rx
        - device 2:Tx -> Channel B, Channel A -> Rx
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
