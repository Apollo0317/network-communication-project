from core.ProtocolStack import ProtocolLayer


class PhyLayer(ProtocolLayer):
    def __init__(self, lower_layer):
        super().__init__(lower_layer)