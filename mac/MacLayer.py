from core import PhySimulationEngine, ProtocolLayer
from mac.protocol import NetworkInterface

class MacLayer(ProtocolLayer):
    def __init__(
        self, 
        lower_layer, 
        simulator:PhySimulationEngine, 
        mac_addr:str=1, 
        mode='node', 
        name:str='zero'
    ):
        super().__init__(lower_layer=lower_layer,name=name, simulator=simulator)
        self.ni= NetworkInterface(mac_addr=mac_addr, mode=mode, name=name)
        self.mac_addr= mac_addr

    def Encapsulate(self, data: tuple):
        if len(data) == 3:
            # Switcher 模式：透传源 MAC
            src_mac, dst_mac, payload = data
        elif len(data) == 2:
            # Node 模式：自动填充本机 MAC 作为源 MAC
            dst_mac, payload = data
            src_mac = self.mac_addr
        else:
            raise ValueError("Invalid data format for MacLayer send")

        return self.ni.encoding(dst_mac=dst_mac, data=payload, src_mac=src_mac)
    
    def Dencapsulate(self, data:bytes)->tuple[int,int,bytes]:
        try:
            data= self.ni.decoding(frame=data)
        except ValueError as e:
            print(f"[MAC Layer {self.name}] Frame decoding error: {e}")
            return None
        else:
            # if self.ni.mode == 'node':
            #     return data[-1]
            return data  # (src_mac, dst_mac, payload)
        