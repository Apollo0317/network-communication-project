"""
define a Node/Host object,

include phy and mac entity while

provide send, recieve and connect interface
"""

import sys

sys.path.append("..")  # 添加父目录到路径

from phy.modulator import Modulator, DeModulator, Cable
from phy.entity import TxEntity, RxEntity, TwistedPair
from mac.entity import MacTxEntity, MacRxEntity
from core.simulator import PhySimulationEngine, SimulationEntity


class Node(SimulationEntity):
    """
    a Node(Host) entity, include phy and mac entity while\n
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

        self.mac_tx = mac_tx
        self.mac_rx = mac_rx
        self.mac_addr = mac_addr
        self.name = name
        self.update = lambda tick: None  # default empty update function

        print(f"Host {name}:{mac_addr} init ok")

    def connect_to(self, twisted_pair: TwistedPair):
        twisted_pair.connect(
            tx_interface=self.mac_tx.phy_interface,
            rx_interface=self.mac_rx.phy_interface,
        )

    def send(self, dst_mac: str, data: bytes):
        self.mac_tx.send(dst_mac=dst_mac, data=data)

    def recv(self) -> None | tuple[str, bytes]:
        return self.mac_rx.recieve()

    def recvall(self) -> list[tuple[str, bytes]]:
        return self.mac_rx.recieve_all()

    def set_event(self, method):
        self.update = method

    def update(self, tick):
        self.update(tick)


class Switcher(SimulationEntity):
    def __init__(
        self,
        simulator: PhySimulationEngine,
        mac_addr: int,
        port_num: int = 3,
        name="switcher",
    ):
        super().__init__(name=name)
        self.port_num = port_num
        self.port_list: list[tuple[MacTxEntity, MacRxEntity]] = []
        self.map: dict[int, int] = {1: 0, 2: 1, 3: 2}
        for i in range(port_num):
            modulator = Modulator(
                scheme="16QAM",
                symbol_rate=1e6,
                sample_rate=50e6,
                fc=2e6,
                power_factor=100,
            )
            demodulator = DeModulator(
                scheme="16QAM",
                symbol_rate=1e6,
                sample_rate=50e6,
                fc=2e6,
                power_factor=100,
            )

            phy_tx = TxEntity(modulator=modulator, name=name + f"{i}_phy_tx")
            phy_rx = RxEntity(demodulator=demodulator, name=name + f"{i}_phy_rx")

            mac_tx = MacTxEntity(
                name=name + f"{i}_mac_tx",
                phy_entity=phy_tx,
                mac_addr=mac_addr,
                mode="switcher",
            )
            mac_rx = MacRxEntity(
                name=name + f"{i}_mac_rx",
                phy_entity=phy_rx,
                mac_addr=mac_addr,
                mode="switcher",
            )

            simulator.register_entity(entity=phy_tx)
            simulator.register_entity(entity=phy_rx)
            simulator.register_entity(entity=mac_tx)
            simulator.register_entity(entity=mac_rx)

            self.port_list.append((mac_tx, mac_rx))

    def connect_to(self, port: int, twisted_pair: TwistedPair):
        mac_tx, mac_rx = self.port_list[port]
        twisted_pair.connect(
            tx_interface=mac_tx.phy_interface, rx_interface=mac_rx.phy_interface
        )
        pass

    def update(self, tick: int):
        if tick % 10 == 0:
            for i in range(self.port_num):
                mac_tx, mac_rx = self.port_list[i]
                result = mac_rx.recieve()
                if result:
                    src_mac, dst_mac, data = result
                    dst_port = self.map.get(dst_mac)
                    if dst_port is not None:
                        dst_mac_tx, _ = self.port_list[dst_port]
                        dst_mac_tx.send(src_mac=src_mac, dst_mac=dst_mac, data=data)
                    else:
                        # TODO: imply flooding algorithm
                        pass


def test():
    simulator = PhySimulationEngine(time_step_us=1, realtime_mode=True)

    node_a = Node(simulator=simulator, mac_addr=1, name="node_a")
    node_b = Node(simulator=simulator, mac_addr=2, name="node_b")
    node_c = Node(simulator=simulator, mac_addr=3, name="node_c")
    switcher = Switcher(simulator=simulator, mac_addr=0, port_num=3, name="switcher")
    simulator.register_entity(entity=node_a)
    simulator.register_entity(entity=node_b)
    simulator.register_entity(entity=node_c)
    simulator.register_entity(entity=switcher)

    # 创建信道
    cable = Cable(
        length=100,
        attenuation=2,
        noise_level=3,
        debug_mode=False,
    )
    print(f"\n{cable}")

    twisted_pair_0 = TwistedPair(cable=cable, simulator=simulator)
    twisted_pair_1 = TwistedPair(cable=cable, simulator=simulator)
    twisted_pair_2 = TwistedPair(cable=cable, simulator=simulator)

    node_a.connect_to(twisted_pair=twisted_pair_0)
    switcher.connect_to(port=0, twisted_pair=twisted_pair_0)
    node_b.connect_to(twisted_pair=twisted_pair_1)
    switcher.connect_to(port=1, twisted_pair=twisted_pair_1)
    node_c.connect_to(twisted_pair=twisted_pair_2)
    switcher.connect_to(port=2, twisted_pair=twisted_pair_2)

    node_a.send(dst_mac=2, data=b"Hi, I am node a!")
    node_b.send(dst_mac=3, data=b"Hi, I am node b!")
    node_c.send(dst_mac=1, data=b"Hi, I am node c!")

    node_a.send(dst_mac=1, data=b"Hi, I am node a!")
    node_b.send(dst_mac=2, data=b"Hi, I am node b!")
    node_c.send(dst_mac=3, data=b"Hi, I am node c!")

    simulator.run(10000)

    print(f"a recv: {node_a.recvall()}")
    print(f"b recv: {node_b.recvall()}")
    print(f"c recv: {node_c.recvall()}")


if __name__ == "__main__":
    test()
