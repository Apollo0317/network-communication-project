from core import ProtocolLayer, PhySimulationEngine
from phy.modulator import Modulator, DeModulator
from phy.entity import TxEntity, RxEntity, TwistedPair
# from phy.Coding import ChannelEncoder


class PhyLayer(ProtocolLayer):
    def __init__(
        self,
        lower_layer: ProtocolLayer,
        coding: bool,
        simulator: PhySimulationEngine,
        name: str = "zero",
    ):
        super().__init__(lower_layer=lower_layer, name=name, simulator=simulator)
        modulator = Modulator(
            scheme="16QAM", symbol_rate=1e6, sample_rate=50e6, fc=2e6, power_factor=100
        )
        demodulator = DeModulator(
            scheme="16QAM", symbol_rate=1e6, sample_rate=50e6, fc=2e6, power_factor=100
        )
        self.tx_entity = TxEntity(modulator=modulator, name=name + "_phy_tx", coding=coding)
        self.rx_entity = RxEntity(demodulator=demodulator, name=name + "_phy_rx", coding=coding)
        simulator.register_entity(entity=self.tx_entity)
        simulator.register_entity(entity=self.rx_entity)
        # self.coding = coding


    def connect_to(self, twisted_pair: TwistedPair):
        twisted_pair.connect(tx_interface=self.tx_entity, rx_interface=self.rx_entity)

    def send_to_phy(self, data):
        self.tx_entity.enqueue_data(data=data)
        pass

    def Encapsulate(self, data):
        """
        imply channel coding
        """
        # if self.use_channel_coding:
        #     return self.channel_encoder.encoding(data=data)
        return data

    def Dencapsulate(self, data):
        """
        imply channel decoding
        """
        # if self.use_channel_coding:
        #     return self.channel_encoder.decoding(data=data)
        return data

    def update(self, tick):
        super().update(tick)
        result = self.rx_entity.get_received_data()
        if result:
            self.handle_data_recieved(data=result)
