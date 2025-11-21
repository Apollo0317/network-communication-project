"""
mac layer entity implementations.
"""

import sys

sys.path.append("..")  # 添加父目录到路径

from phy.modulator import Modulator, DeModulator, Cable
from phy.entity import TxEntity, RxEntity, ChannelEntity
from core.simulator import PhySimulationEngine, SimulationEntity
from mac.protocol import NetworkInterface


class MacTxEntity(SimulationEntity):
    def __init__(self, name: str, phy_entity: TxEntity, mac_addr: str):
        super().__init__(name)
        self.phy_interface = phy_entity
        self.mac_addr = mac_addr
        self.ni = NetworkInterface(mac_addr=mac_addr, name=name)

    def send(self, dst_mac: str, data: bytes):
        print(f"dst_mac:{dst_mac} data:{data}")
        frame = self.ni.encoding(dst_mac=dst_mac, data=data)
        self.phy_interface.enqueue_data(data=frame)

    def update(self, tick: int):
        pass


class MacRxEntity(SimulationEntity):
    def __init__(self, name: str, phy_entity: RxEntity, mac_addr: str):
        super().__init__(name)
        self.phy_interface = phy_entity
        self.mac_addr = mac_addr
        self.ni = NetworkInterface(mac_addr=mac_addr, name=name)
        self.rx_queue: list[tuple[int, bytes]] = []

    def update(self, tick):
        frame = self.phy_interface.get_received_data()
        if frame:
            try:
                src_mac, data = self.ni.decoding(frame)
            except ValueError as e:
                print(f"[{self.name}] Error decoding frame: {e}")
                return
            else:
                self.rx_queue.append((src_mac, data))

    def recieve(self) -> tuple[int, bytes] | None:
        if self.rx_queue:
            return self.rx_queue.pop(0)
        return None

    def recieve_all(self) -> list[tuple[int, bytes]]:
        results = self.rx_queue.copy()
        self.rx_queue.clear()
        return results

        pass


def test():
    simulator = PhySimulationEngine(time_step_us=1, realtime_mode=False)

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

    # 计算传播延迟（ticks）
    propagation_delay_s = cable.get_propagation_delay()
    time_step_us = 1.0
    propagation_delay_ticks = int(propagation_delay_s / (time_step_us * 1e-6))

    tx_phy = TxEntity(name="test_tx_phy", modulator=modulator)
    rx_phy = RxEntity(name="test_rx_phy", demodulator=demodulator)
    channel = ChannelEntity(
        cable=cable,
        propagation_delay_ticks=propagation_delay_ticks,
        name="test_channel",
    )
    tx_phy.connect_to_channel(channel)
    channel.connect_receiver(rx_phy)

    mac_tx = MacTxEntity(name="test_mac_tx", phy_entity=tx_phy, mac_addr=1)
    mac_rx = MacRxEntity(name="test_mac_rx", phy_entity=rx_phy, mac_addr=2)

    simulator.register_entity(channel)
    simulator.register_entity(tx_phy)
    simulator.register_entity(rx_phy)
    simulator.register_entity(mac_tx)
    simulator.register_entity(mac_rx)

    mac_tx.send(dst_mac=2, data=b"Hello, this is a test message.")
    mac_tx.send(dst_mac=2, data=b"Hello, this is a test message.")
    mac_tx.send(dst_mac=2, data=b"Hello, this is a test message.")

    simulator.run(duration_ticks=10000)

    result = mac_rx.recieve_all()
    print(f"Received Result: {result}")

    pass


if __name__ == "__main__":
    test()
