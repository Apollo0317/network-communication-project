from core.ProtocolStack import ProtocolLayer
from core.simulator import PhySimulationEngine
from phy.modulator import Modulator, DeModulator
from phy.entity import TxEntity, RxEntity, TwistedPair


class PhyLayer(ProtocolLayer):
    def __init__(self, lower_layer:ProtocolLayer, simulator:PhySimulationEngine, name:str='zero'):
        super().__init__(lower_layer=lower_layer, name=name, simulator=simulator)
        modulator = Modulator(
            scheme="16QAM", symbol_rate=1e6, sample_rate=50e6, fc=2e6, power_factor=100
        )
        demodulator = DeModulator(
            scheme="16QAM", symbol_rate=1e6, sample_rate=50e6, fc=2e6, power_factor=100
        )
        self.tx_entity= TxEntity(modulator=modulator, name=name+ "_phy_tx")
        self.rx_entity= RxEntity(demodulator=demodulator, name=name+"_phy_rx")
        simulator.register_entity(entity=self.tx_entity)
        simulator.register_entity(entity=self.rx_entity)

    def connect_to(self, twisted_pair: TwistedPair):
        twisted_pair.connect(
            tx_interface=self.tx_entity,
            rx_interface=self.rx_entity
        )

    def send_to_phy(self, data):
        self.tx_entity.enqueue_data(data=data)
        pass

    def Encapsulate(self, data):
        """
        should never be called in phy layer!
        """
        return data
        pass

    def Dencapsulate(self, data):
        """
        should never be called in phy layer!
        """
        return data
    
    def update(self, tick):
        super().update(tick)
        result= self.rx_entity.get_received_data()
        if result:
            self.handle_data_recieved(data=result)
        

    


