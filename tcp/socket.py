from core import SimulationEntity, ProtocolLayer, PhySimulationEngine
from tcp.TransportLayer import TransportSession, TransportLayer_GBN
from typing import Optional, Self


class socket:
    def __init__(self, tcp_layer: TransportLayer_GBN):
        self.locol_port = 8080
        self.tcp_layer = tcp_layer
        self.greeting_data = b"Hi"
        self.session: TransportSession = None
        self.is_listening = False
        self.listen_num = 0
        self.connection_num = 0
        self.established_sessions: list[TransportSession] = []
        self.debug_mode = False

    def setmode(self, mode: str):
        """
        mode: 'debug' 开启高层时间线输出，其他关闭
        """
        if mode == "debug":
            self.debug_mode = True
        else:
            self.debug_mode = False

    def _timeline_log(self, action: str, extra: str = ""):
        if not self.debug_mode:
            return
        tick = self.tcp_layer.simulator.current_tick
        node_name = getattr(self.tcp_layer, "name", "tcp")
        peer = ""
        if self.session is not None:
            peer = f" peer={self.session.remote_mac}:{self.session.remote_port}"
        print(
            f"[TICK {tick}][{node_name}] {action} "
            f"sock_port={self.locol_port}{peer} {extra}"
        )

    def bind(self, port: int):
        self.locol_port = port
        self._timeline_log("BIND", f"-> port={port}")

    def connect(self, dst_mac: int, dst_port: int):
        self.session = self.tcp_layer.send(
            (self.locol_port, dst_mac, dst_port, self.greeting_data)
        )
        self._timeline_log(
            "CONNECT",
            f"to={dst_mac}:{dst_port} session={self.session}",
        )

    def listen(self, num: int = 5):
        self.is_listening = True
        self.listen_num = num
        self._timeline_log("LISTEN", f"backlog={num}")

    def accept(self) -> Optional[Self]:
        if not self.is_listening:
            raise LookupError("accept before listen")
        my_sessions = [
            s
            for s in self.tcp_layer.session2data.keys()
            if s.local_port == self.locol_port
        ]
        for session in my_sessions:
            if session not in self.established_sessions:
                self.established_sessions.append(session)
                buffer = self.tcp_layer.session2data.get(session, b"")
                if buffer.startswith(self.greeting_data):
                    buffer = buffer[len(self.greeting_data) :]
                    self.tcp_layer.session2data[session] = buffer
                new_socket = socket(tcp_layer=self.tcp_layer)
                new_socket.bind(self.locol_port)
                new_socket.session = session
                # 继承 debug 模式，方便继续打时间线
                new_socket.debug_mode = self.debug_mode

                self._timeline_log(
                    "ACCEPT",
                    f"new_session={session} "
                    f"peer={session.remote_mac}:{session.remote_port} "
                    f"pending_data_len={len(buffer)}",
                )
                return new_socket

        self._timeline_log("ACCEPT", "no_pending_session")
        return None

    def send(self, data: bytes):
        if not self.session:
            raise LookupError("send before connection: tcp is based on connection!")
        dst_mac = self.session.remote_mac
        dst_port = self.session.remote_port
        self.tcp_layer.send((self.locol_port, dst_mac, dst_port, data))

        # 只展示前 16B 预览，避免刷屏
        preview = data[:16]
        self._timeline_log(
            "SEND",
            f"len={len(data)} preview={preview!r}",
        )


    def recv(self, length: int = 128) -> Optional[bytes]:
        """
        接收最多 length 字节数据（非阻塞）
        - 有数据时：返回 min(len(buffer), length) 字节
        - 无数据时：返回 None
        """
        if self.is_listening:
            raise LookupError(
                "listening socket cannot recv data directly, use accept() first."
            )
        if not self.session:
            raise LookupError("recv before connection: tcp is based on connection!")

        if self.session not in self.tcp_layer.session2data:
            self._timeline_log("RECV", "no_data")
            return None
        
        buffer = self.tcp_layer.session2data[self.session]
        if not buffer:
            self._timeline_log("RECV", "empty_buffer")
            return None
        
        data = buffer[:length]
        remaining = buffer[length:]
        
        if remaining:
            self.tcp_layer.session2data[self.session] = remaining
        else:
            del self.tcp_layer.session2data[self.session]
        
        preview = data[:64]
        self._timeline_log(
            "RECV",
            f"len={len(data)} preview={preview!r}",
        )
        return data

