import abc
from typing import Self,  Optional
from collections import deque
from core.simulator import SimulationEntity, PhySimulationEngine

class ProtocolLayer(SimulationEntity):
    def __init__(self, lower_layer, simulator:PhySimulationEngine, name:str='zero'):
        super().__init__(name=name)
        self.tx_queue= deque()
        self.rx_queue= deque()
        self.lower_layer:Self= lower_layer
        self.upper_layer:Optional[Self]= None
        if self.lower_layer:
            self.lower_layer.upper_layer= self
        self.simulator= simulator
        simulator.register_entity(self)
        pass
        
    def handle_data_recieved(self, data):
        """
        upper layer interface for recv
        """
        dencapsulated_data= self.Dencapsulate(data=data)
        if dencapsulated_data is None:
            #print(f"[{self.name} Layer] Dencapsulation returned None, dropping packet.")
            return
        if self.upper_layer:
            self.upper_layer.handle_data_recieved(data=dencapsulated_data)
        else:
            self.rx_queue.append(dencapsulated_data)
        pass
    
    def recv(self):
        """
        recv method for top(app) layer
        """
        if self.rx_queue:
            return self.rx_queue.popleft()

    def send(self, data):
        """
        upper layer interface for send
        """
        encapsulated_data= self.Encapsulate(data=data)
        if self.lower_layer:
            self.lower_layer.send(data=encapsulated_data)
        else:
            self.send_to_phy(data=encapsulated_data)

    def send_to_phy(self, data:bytes):
        """
        send method for the phy layer
        """
        pass

    def Encapsulate(self, data):
        """
        Encapsulate upper layer data to current layer data\n
        """
        pass
    
    def Dencapsulate(self, data):
        """
        Dencapsulate lower layer data to current layer data\n
        """
        pass

    def update(self, tick):
        pass

