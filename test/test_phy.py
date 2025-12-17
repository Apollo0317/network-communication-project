import sys
sys.path.append("..")  # 添加父目录到路径
from phy import PhyLayer, TwistedPair, Cable
from core import PhySimulationEngine, SimulationEntity
from utils import generate_random_data, diff
from typing import Optional
import random

class TestNode(SimulationEntity):
    def __init__(self, simulator:PhySimulationEngine, name:str='zero'):
        super().__init__(name=name)
        self.phy_layer= PhyLayer(lower_layer=None, coding=True, simulator=simulator, name=name)
        self.socket_layer= self.phy_layer
        self.name= name
        simulator.register_entity(self)

    def send(self, data:bytes):
        self.socket_layer.send(data)


    def recv(self)->Optional[bytes]:
        result = self.socket_layer.recv()
        if result:
            return result
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


def test_p2p_communication():
    simulator= PhySimulationEngine()
    simulator.set_debug(False)

    node_a= TestNode(simulator=simulator, name='NodeA')
    node_b= TestNode(simulator=simulator, name='NodeB')

    test_msg= generate_random_data(length= 1024*16)  # 64KB random data
    print(f'NodeA sending {len(test_msg)} bytes:\n')

    cable = Cable(
        length= 100,
        attenuation= 3.5,
        noise_level= 4,
        debug_mode= False,
    )

    tp = TwistedPair(
        cable= cable,
        simulator= simulator,
    )

    node_a.connect_to(tp)
    node_b.connect_to(tp)

    node_a.send(test_msg)

    simulator.run(duration_ticks=100000)

    received_msgs= node_b.recv()

    print(f'NodeB received {len(received_msgs)} bytes:\n')

    diff_bytes= diff(test_msg, received_msgs)
    print(f'Different bytes: {diff_bytes}')

    assert received_msgs is not None
    assert received_msgs == test_msg

if __name__ == '__main__':
    test_p2p_communication()
