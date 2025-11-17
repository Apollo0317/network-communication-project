
"""
Provides Node class for MAC layer operations.

Includes methods for sending and receiving data packets
"""

from phy.modulator import Modulator, DeModulator, Cable

class Node:
    def __init__ (self, 
        modulator: Modulator, 
        demodulator: DeModulator,
        mac_addr: str,
        host_name: str,
        cable: Cable
    ):
        self.modulator= modulator
        self.demodulator= demodulator
        self.mac_addr= mac_addr
        self.host_name= host_name
        self.cable= cable
        print(f'{host_name}:{mac_addr} init ok')
    
    def send(self, data:bytes):
        signal= self.modulator.modulate(data=data)

    def recv(self)->bytes:
        pass


class Switcher(Node):
    def __init__(self):
        super().__init__()
        pass


    

def test():
    pass

if __name__ == "__main__":
    test()
