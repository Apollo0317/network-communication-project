"""
define a Node/Host object,

include phy and mac entity while

provide send, recieve and connect interface
"""

import sys

sys.path.append("..")  # 添加父目录到路径

from phy.modulator import Modulator, DeModulator, Cable
from phy.entity import TxEntity, RxEntity, ChannelEntity
from mac.entity import MacTxEntity, MacRxEntity
from core.simulator import PhySimulationEngine, SimulationEntity

import copy


class Node(SimulationEntity):
    """
    a Node\\Host entity, include phy and mac entity while\n
    provide send, recieve and connect interface
    """

    def __init__(
        self, simulator: PhySimulationEngine, mac_addr: str, name: str = "zero"
    ):
        """
        build a Node from scratch
        """
        modulator = Modulator(
            scheme="16QAM", symbol_rate=1e6, sample_rate=50e6, fc=2e6, power_factor=100
        )
        demodulator = DeModulator(
            scheme="16QAM", symbol_rate=1e6, sample_rate=50e6, fc=2e6, power_factor=100
        )

        phy_tx = TxEntity(modulator=modulator, name=name + "_phy_tx")
        phy_rx = RxEntity(demodulator=demodulator, name=name + "_phy_rx")

        mac_tx = MacTxEntity(
            name=name + "_mac_tx", phy_entity=phy_tx, mac_addr=mac_addr
        )
        mac_rx = MacRxEntity(
            name=name + "_mac_rx", phy_entity=phy_rx, mac_addr=mac_addr
        )

        simulator.register_entity(entity=phy_tx)
        simulator.register_entity(entity=phy_rx)
        simulator.register_entity(entity=mac_tx)
        simulator.register_entity(entity=mac_rx)

        self.phy_tx = phy_tx
        self.phy_rx = phy_rx
        self.mac_tx = mac_tx
        self.mac_rx = mac_rx
        self.mac_addr = mac_addr
        self.name = name

        print(f"Host {name}:{mac_addr} init ok")

    def connect_to(self, channel: ChannelEntity):
        self.mac_tx.phy_interface.connect_to_channel(channel=channel)
        self.tx_channel = channel

    def connect_from(self, channel: ChannelEntity):
        channel.connect_receiver(rx=self.mac_rx.phy_interface)
        self.rx_channel = channel

    def send(self, dst_mac: str, data: bytes):
        self.mac_tx.send(dst_mac=dst_mac, data=data)

    def recv(self) -> None | tuple[str, bytes]:
        return self.mac_rx.recieve()

    def recvall(self) -> list[tuple[str, bytes]]:
        return self.mac_rx.recieve_all()


def test():
    simulator = PhySimulationEngine(time_step_us=1)
    node_a = Node(simulator=simulator, mac_addr=1, name="node_a")
    node_b = Node(simulator=simulator, mac_addr=2, name="node_b")

    # 创建信道
    cable = Cable(
        length=100,
        attenuation=2,
        noise_level=3,
        debug_mode=False,
    )
    print(f"\n{cable}")

    channel_tx = ChannelEntity(
        cable=cable,
        name="channel_a",
    )
    channel_rx = ChannelEntity(
        cable=cable,
        name="channel_b",
    )

    simulator.register_entity(channel_tx)
    simulator.register_entity(channel_rx)

    node_a.connect_to(channel_tx)
    node_b.connect_from(channel_tx)

    node_b.connect_to(channel_rx)
    node_a.connect_from(channel_rx)

    node_a.send(dst_mac=2, data=b"Hi, I am node a!")
    node_b.send(dst_mac=1, data=b"Hi, I am node b!")

    simulator.run(10000)

    print(f"a recv: {node_a.recv()}")
    print(f"b recv: {node_b.recv()}")


if __name__ == "__main__":
    test()
