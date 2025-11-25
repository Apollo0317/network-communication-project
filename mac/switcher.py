
import sys
sys.path.append("..")  # 添加父目录到路径
from core import PhySimulationEngine, SimulationEntity, ProtocolLayer
from phy import PhyLayer, TwistedPair
from mac.MacLayer import MacLayer

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
        self.socket_list:list[ProtocolLayer] = []
        self.port_list: list[PhyLayer] = []
        self.map: dict[int, int] = {1: 0, 2: 1, 3: 2}
        for i in range(port_num):
            phy_layer = PhyLayer(lower_layer=None, coding=True, simulator=simulator, name=name)
            mac_layer = MacLayer(
                lower_layer=phy_layer,
                simulator=simulator,
                mode='switch',
                mac_addr=mac_addr,
                name=name,
            )
            self.socket_list.append(mac_layer)
            self.port_list.append(phy_layer)
        simulator.register_entity(self)



    def connect_to(self, port: int, twisted_pair: TwistedPair):
        phy_layer = self.port_list[port]
        twisted_pair.connect(
            tx_interface=phy_layer.tx_entity, rx_interface=phy_layer.rx_entity
        )
        pass

    def update(self, tick: int):
        if tick % 10 == 0:
            for i in range(self.port_num):
                socket_layer = self.socket_list[i]
                result = socket_layer.recv()
                if result:
                    src_mac, dst_mac, data = result
                    dst_port = self.map.get(dst_mac)
                    if dst_port is not None:
                        dst_port_socket_layer = self.socket_list[dst_port]
                        dst_port_socket_layer.send((src_mac,dst_mac, data))
                    else:
                        # TODO: imply flood algo
                        pass