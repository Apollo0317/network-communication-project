from core import SimulationEntity, ProtocolLayer, PhySimulationEngine
from tcp.TransportLayer import TransportSession, TransportLayer_GBN
from typing import Optional, Self

class socket:
    def __init__(self, tcp_layer:TransportLayer_GBN):
        self.locol_port= 8080
        self.tcp_layer= tcp_layer
        self.greeting_data=b'Hi'
        self.session:TransportSession= None
        self.is_listening= False
        self.listen_num= 0
        self.connection_num= 0
        self.established_sessions: list[TransportSession]= []

    def bind(self, port:int):
        self.locol_port= port

    def connect(self, dst_mac:int, dst_port:int):
        self.session= self.tcp_layer.send((self.locol_port, dst_mac, dst_port, self.greeting_data))

    def listen(self, num:int=5):
        self.is_listening= True
        self.listen_num= num
        pass

    def accept(self)-> Self:
        if not self.is_listening:
            raise LookupError("accept before listen")
        my_sessions= [s for s in self.tcp_layer.session2data.keys() if s.local_port==self.locol_port]
        for session in my_sessions:
            if session not in self.established_sessions:
                self.established_sessions.append(session)
                buffer=self.tcp_layer.session2data.get(session)
                # 去掉 greeting_data（这里假设就是 b'Hi'，长度为 2）
                if buffer.startswith(self.greeting_data):
                    buffer = buffer[len(self.greeting_data):]
                    self.tcp_layer.session2data[session] = buffer  # 关键：写回去
                new_socket= socket(tcp_layer=self.tcp_layer)
                new_socket.bind(self.locol_port)
                new_socket.session= session
                return new_socket
                
        pass

    def send(self, data:bytes):
        if not self.session:
            raise LookupError("send before connection: tcp is based on connection!")
        dst_mac= self.session.remote_mac
        dst_port= self.session.remote_port
        self.tcp_layer.send((self.locol_port, dst_mac, dst_port, data))
        #print(f'send {data} to {dst_mac}:{dst_port} from port {self.locol_port}')
    
    def recv(self)-> Optional[bytes]:
        if self.is_listening:
            raise LookupError("listening socket cannot recv data directly, use accept() to get connected socket first.")
        if not self.session:
            raise LookupError("recv before connection: tcp is based on connection!")
        if self.session in self.tcp_layer.session2data.keys():
            data= self.tcp_layer.session2data[self.session]
            del self.tcp_layer.session2data[self.session]
            return data
        else:
            return None
    
    
