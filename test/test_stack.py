# test/test_stack.py
import sys
sys.path.append("..")  # 添加父目录到路径
from phy.PhyLayer import PhyLayer
from phy.entity import TwistedPair
from phy.cable import Cable
from mac.MacLayer import MacLayer
from mac.switcher import Switcher
from core.simulator import PhySimulationEngine, SimulationEntity
from core.ProtocolStack import ProtocolLayer
# import rich.live

class TestNode(SimulationEntity):
    def __init__(self, simulator:PhySimulationEngine, mac_addr:str, name:str='zero', mode:str='node'):
        super().__init__(name=name)
        self.phy_layer= PhyLayer(lower_layer=None, simulator=simulator, name=name)
        self.mac_layer= MacLayer(
                    lower_layer=self.phy_layer, 
                    simulator=simulator,mode=mode, 
                    mac_addr=mac_addr, name=name
                )
        self.socket_layer= self.mac_layer
        self.mac_addr= mac_addr
        self.name= name
        simulator.register_entity(self)

    def connect_to(self, twisted_pair:TwistedPair):
        twisted_pair.connect(
            tx_interface=self.phy_layer.tx_entity,
            rx_interface=self.phy_layer.rx_entity
        )

    def send(self, dst_mac:int, data:bytes):
        self.socket_layer.send(data=(dst_mac, data))

    def recv(self):
        data= self.socket_layer.recv()
        return data

    def recv_all(self):
        data_list= []
        while True:
            data= self.socket_layer.recv()
            if data is None:
                break
            data_list.append(data)
        return data_list
    

def test_mac_phy_integration():
    simulator= PhySimulationEngine(time_step_us=1)
    node1= TestNode(simulator=simulator, mac_addr=1, name='node1')
    node2= TestNode(simulator=simulator, mac_addr=2, name='node2')
    node3= TestNode(simulator=simulator, mac_addr=3, name='node3')
    switcher= Switcher(simulator=simulator, mac_addr=0, port_num=3, name='switcher')

    # 创建信道
    cable = Cable(
        length=100,
        attenuation=4,
        noise_level=3.5,
        debug_mode=False,
    )
    print(f"\n{cable}")
    tp1= TwistedPair(cable=cable, simulator=simulator)
    tp2= TwistedPair(cable=cable, simulator=simulator)
    tp3= TwistedPair(cable=cable, simulator=simulator)

    node1.connect_to(tp1)
    switcher.connect_to(port=0, twisted_pair=tp1)
    node2.connect_to(tp2)
    switcher.connect_to(port=1, twisted_pair=tp2)
    node3.connect_to(tp3)
    switcher.connect_to(port=2, twisted_pair=tp3)

    test_data= b'Hello, this is a test message.'
    node1.send(2, data=b'Hello, this is Node 1.')
    node2.send(3, data=b'Hello, this is Node 2.')
    node3.send(1, data=b'Hello, this is Node 3.')

    simulator.run(duration_ticks=1000)

    node_1_received= node1.recv_all()
    node_2_received= node2.recv_all()
    node_3_received= node3.recv_all()

    print(f"\n[Node 1] Received: {node_1_received}")
    print(f"[Node 2] Received: {node_2_received}")
    print(f"[Node 3] Received: {node_3_received}")

if __name__ == "__main__":
    test_mac_phy_integration()