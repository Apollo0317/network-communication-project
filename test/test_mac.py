import sys
sys.path.append("..")  # 添加父目录到路径
from phy import PhyLayer, TwistedPair, Cable
from mac import MacLayer, Switcher
from core import PhySimulationEngine, SimulationEntity
from typing import Optional

class TestNode(SimulationEntity):
    def __init__(self, simulator:PhySimulationEngine, mac_addr:str, name:str='zero', mode:str='node'):
        super().__init__(name=name)
        self.phy_layer= PhyLayer(lower_layer=None, coding=True, simulator=simulator, name=name)
        self.mac_layer= MacLayer(
                    lower_layer=self.phy_layer, 
                    simulator=simulator,mode=mode, 
                    mac_addr=mac_addr, name=name
                )
        self.socket_layer= self.mac_layer
        self.mac_addr= mac_addr
        self.name= name
        simulator.register_entity(self)

    def send(self, dst_mac:int, data:bytes):
        self.socket_layer.send((self.mac_addr, dst_mac, data))


    def recv(self)->Optional[bytes]:
        result = self.socket_layer.recv()
        if result:
            _, _, data = result
            return data
        return None

    def recvall(self)->Optional[list[bytes]]:
        results= []
        result= self.recv()
        while result:
            results.append(result)
            result= self.recv()
        return results

    def connect_to(self, twisted_pair:TwistedPair):
        twisted_pair.connect(
            tx_interface=self.phy_layer.tx_entity,
            rx_interface=self.phy_layer.rx_entity
        )
        print(f'[{self.name}] connected to {twisted_pair}')
    
    def update(self, tick):
        super().update(tick)

def test_two_clients_communicate_via_switcher():
    simulator = PhySimulationEngine(time_step_us=1)

    node1 = TestNode(simulator=simulator, mac_addr=1, name='node1')
    node2 = TestNode(simulator=simulator, mac_addr=2, name='node2')
    switcher = Switcher(simulator=simulator, mac_addr=0, port_num=2, name='switcher')

    cable = Cable(
        length=100,
        attenuation=4,
        noise_level=4,
        debug_mode=False,
    )
    print(f"\n{cable}")
    tp1 = TwistedPair(cable=cable, simulator=simulator, ID=0)
    tp2 = TwistedPair(cable=cable, simulator=simulator, ID=1)

    node1.connect_to(tp1)
    switcher.connect_to(port=0, twisted_pair=tp1)
    node2.connect_to(tp2)
    switcher.connect_to(port=1, twisted_pair=tp2)

    test_msg=  b"Hello from Node 1 to Node 2 via Switcher"
    print(f"\n[node1] Sending message to [node2]: {test_msg}")

    node1.send(dst_mac=2, data=test_msg)

    simulator.run(duration_ticks=2000)

    received_msgs= node2.recvall()

    print(f"\n[node2] Received messages: {received_msgs}")


    assert received_msgs is not None, "[node2] No messages received"
    assert len(received_msgs) == 1, f"[node2] Expected 1 message, got {len(received_msgs)}"
    assert received_msgs[0] == test_msg, f"[node2] Message content mismatch"

    print("\n=== Mac Test Passed ===")

def test_switcher_mac_learning():
    simulator = PhySimulationEngine(time_step_us=1)

    node1 = TestNode(simulator=simulator, mac_addr=1, name='node1')
    node2 = TestNode(simulator=simulator, mac_addr=2, name='node2')
    switcher = Switcher(simulator=simulator, mac_addr=0, port_num=2, name='switcher')

    cable = Cable(
        length=100,
        attenuation=4,
        noise_level=3,
        debug_mode=False,
    )
    print(f"\n{cable}")
    tp1 = TwistedPair(cable=cable, simulator=simulator, ID=0)
    tp2 = TwistedPair(cable=cable, simulator=simulator, ID=1)

    node1.connect_to(tp1)
    switcher.connect_to(port=0, twisted_pair=tp1)
    node2.connect_to(tp2)
    switcher.connect_to(port=1, twisted_pair=tp2)

    print(f"\nSwitcher MAC Table: {switcher.map}")

    test_msg1=  b"Message 1 from Node 1 to Node 2"
    test_msg2=  b"Message 2 from Node 2 to Node 1"

    print(f"\n[node1] Sending message to [node2]: {test_msg1}")
    node1.send(dst_mac=2, data=test_msg1)

    simulator.run(duration_ticks=2000)

    received_msgs_node2= node2.recvall()
    print(f"\n[node2] Received messages: {received_msgs_node2}")

    print(f"\n[node2] Sending message to [node1]: {test_msg2}")
    node2.send(dst_mac=1, data=test_msg2)

    simulator.run(duration_ticks=2000)

    received_msgs_node1= node1.recvall()
    print(f"\n[node1] Received messages: {received_msgs_node1}")


    print(f"\nSwitcher MAC Table: {switcher.map}")
    assert switcher.map.get(1) == 0, "Switcher did not learn MAC 1 correctly"
    assert switcher.map.get(2) == 1, "Switcher did not learn MAC 2 correctly"

    print("\n=== Switcher MAC Learning Test Passed ===")

def test_three_nodes_communicate_via_switcher():
    simulator = PhySimulationEngine(time_step_us=1)

    node1 = TestNode(simulator=simulator, mac_addr=1, name='node1')
    node2 = TestNode(simulator=simulator, mac_addr=2, name='node2')
    node3 = TestNode(simulator=simulator, mac_addr=3, name='node3')
    switcher = Switcher(simulator=simulator, mac_addr=0, port_num=3, name='switcher')

    cable = Cable(
        length=100,
        attenuation=4,
        noise_level=3,
        debug_mode=False,
    )
    print(f"\n{cable}")
    tp1 = TwistedPair(cable=cable, simulator=simulator, ID=0)
    tp2 = TwistedPair(cable=cable, simulator=simulator, ID=1)
    tp3 = TwistedPair(cable=cable, simulator=simulator, ID=2)

    node1.connect_to(tp1)
    switcher.connect_to(port=0, twisted_pair=tp1)
    node2.connect_to(tp2)
    switcher.connect_to(port=1, twisted_pair=tp2)
    node3.connect_to(tp3)
    switcher.connect_to(port=2, twisted_pair=tp3)

    test_msg1=  b"Hello from Node 1 to Node 2"
    test_msg2=  b"Hello from Node 3 to Node 2"

    print(f"\n[node1] Sending message to [node2]: {test_msg1}")
    node1.send(dst_mac=2, data=test_msg1)

    print(f"\n[node3] Sending message to [node2]: {test_msg2}")
    node3.send(dst_mac=2, data=test_msg2)

    simulator.run(duration_ticks=3000)

    received_msgs_node2= node2.recvall()
    print(f"\n[node2] Received messages: {received_msgs_node2}")

    assert received_msgs_node2 is not None, "[node2] No messages received"
    assert len(received_msgs_node2) == 2, f"[node2] Expected 2 messages, got {len(received_msgs_node2)}"
    assert test_msg1 in received_msgs_node2, "[node2] Message from Node 1 missing"
    assert test_msg2 in received_msgs_node2, "[node2] Message from Node 3 missing"


if __name__ == "__main__":
    test_switcher_mac_learning()