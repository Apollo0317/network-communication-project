import struct
from core import SimulationEntity, PhySimulationEngine, ProtocolLayer
from typing import Optional, Dict, Tuple, OrderedDict
from mac import MacLayer

# 协议头格式: Type(1B) | Src_Port(2B) | Dst_Port(2B) |Seq(2B) | Ack(2B) | RWND(2B) (Optional, only for ACK in SR)
# Type: 0=DATA, 1=ACK
HEADER_FMT = "!BHHHH"
HEAD_FMT_ACK = "!BHHHHH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
ACK_HEADER_SIZE = struct.calcsize(HEAD_FMT_ACK)
TYPE_DATA = 0
TYPE_ACK = 1

# 配置参数
TIMEOUT_TICKS = 128  # 超时时间
WINDOW_SIZE = 12  # 发送窗口大小
MSS = 1024  # Maximum Segment Size: 每个包的最大负载长度


class TransportSession:
    """维护针对特定目标MAC的传输状态"""

    def __init__(
        self, local_mac: int, local_port: int, remote_mac: int, remote_port: int
    ):
        self.local_mac = local_mac
        self.local_port = local_port
        self.remote_mac = remote_mac
        self.remote_port = remote_port
        self.seq_out = 0  # 下一个发送的序列号
        self.seq_in = 0  # 期望接收的序列号
        # 已发送但未确认的数据包: seq -> (tick_sent, packet_bytes)
        self.unacked_packets: OrderedDict[int, Tuple[int, bytes]] = {}
        self.send_buffer = b""  # [新增] 发送缓冲区
        self.recieve_buffer: list[tuple[int, bytes]] = []
        self.tx_window_size = WINDOW_SIZE
        self.rx_window_size = WINDOW_SIZE

    def __str__(self):
        return f"[session:({self.local_mac}:{self.local_port}) -> ({self.remote_mac}:{self.remote_port})]"


class TransportLayer_GBN(ProtocolLayer):
    def __init__(
        self,
        lower_layer: MacLayer,
        simulator: PhySimulationEngine = PhySimulationEngine(),
        name: str = "transport_layer",
    ):
        super().__init__(lower_layer=lower_layer, simulator=simulator, name=name)
        # 使用 MAC 地址映射会话状态
        self.sessions: Dict[tuple[int, int, int, int], TransportSession] = {}
        self.session2data: Dict[TransportSession, bytes] = {}
        # debug 开关
        self.debug_mode: bool = True

    # 简单的调试打印工具
    def debug_log(self, msg: str):
        if self.debug_mode:
            print(f"[TICK {self.simulator.current_tick}][{self.name}] {msg}")

    def _get_session(
        self, local_mac: int, local_port: int, remote_mac: int, remote_port: int
    ) -> TransportSession:
        key = (local_mac, local_port, remote_mac, remote_port)
        if key not in self.sessions.keys():
            self.sessions[key] = TransportSession(local_mac, local_port, remote_mac, remote_port)
            self.debug_log(
                f"NEW SESSION {local_mac}:{local_port} -> {remote_mac}:{remote_port}"
            )
        return self.sessions[key]

    def Encapsulate(self, session: TransportSession, data: bytes) -> Optional[tuple]:
        """
        封装逻辑：添加 Seq/Ack 头，并缓存用于重传
        """
        header = struct.pack(
            HEADER_FMT,
            TYPE_DATA,
            session.local_port,
            session.remote_port,
            session.seq_out,
            session.seq_in,
        )
        packet = header + data

        session.unacked_packets[session.seq_out] = (self.simulator.current_tick, packet)
        self.debug_log(
            f"SEND DATA seq={session.seq_out} len={len(data)} "
            f"{session.local_mac}:{session.local_port} -> "
            f"{session.remote_mac}:{session.remote_port} "
            f"unacked={list(session.unacked_packets.keys())}"
        )

        session.seq_out += 1
        return (session.remote_mac, packet)

    def Dencapsulate(self, data: tuple) -> Optional[tuple[bytes, int]]:
        """
        解封装逻辑：处理 ACK，校验 Seq
        Args:
            data: (src_mac, dst_mac, frame) 来自 MAC 层
        """
        if not isinstance(data, tuple) or len(data) != 3:
            return None

        src_mac, dst_mac, frame = data
        if len(frame) < HEADER_SIZE:
            return None

        msg_type, src_port, dst_port, seq, ack = struct.unpack(
            HEADER_FMT, frame[:HEADER_SIZE]
        )
        payload: bytes = frame[HEADER_SIZE:]

        session = self._get_session(
            local_mac=dst_mac,
            local_port=dst_port,
            remote_mac=src_mac,
            remote_port=src_port,
        )

        # 1. Reliable Transport (ACK处理)
        acked_seqs = [s for s in session.unacked_packets if s < ack]
        if acked_seqs:
            for s in acked_seqs:
                del session.unacked_packets[s]
            self.debug_log(
                f"RECV ACK ack={ack} from {src_mac}:{src_port} "
                f"session={session} remaining_unacked={list(session.unacked_packets.keys())}"
            )
            self._flush_send_buffer(session)

        if msg_type == TYPE_ACK:
            return None  # 纯 ACK 包不向上传递

        elif msg_type == TYPE_DATA:
            self.debug_log(
                f"RECV DATA seq={seq} len={len(payload)} "
                f"{src_mac}:{src_port} -> {dst_mac}:{dst_port} "
                f"expect_seq_in={session.seq_in}"
            )
            # 接收数据逻辑
            if seq == session.seq_in:
                session.seq_in += 1
                self._send_ack(session)

                if session not in self.session2data.keys():
                    self.session2data[session] = payload
                else:
                    self.session2data[session] += payload
                self.debug_log(
                    f"APPEND PAYLOAD to session {session} "
                    f"total_len={len(self.session2data[session])}"
                )
                return None

            elif seq < session.seq_in:
                # 重复包，重发 ACK
                self.debug_log(
                    f"DUP DATA seq={seq} < seq_in={session.seq_in}, resend ACK"
                )
                self._send_ack(session)
                return None
            else:
                # 乱序包
                self.debug_log(
                    f"OUT-OF-ORDER DATA seq={seq} > seq_in={session.seq_in}, drop & resend ACK"
                )
                self._send_ack(session)
                return None

    def _send_ack(self, session: TransportSession):
        """发送纯 ACK 包"""
        header = struct.pack(
            HEADER_FMT,
            TYPE_ACK,
            session.local_port,
            session.remote_port,
            0,
            session.seq_in,
        )
        self.debug_log(
            f"SEND ACK ack={session.seq_in} "
            f"{session.local_mac}:{session.local_port} -> "
            f"{session.remote_mac}:{session.remote_port}"
        )
        if self.lower_layer:
            self.lower_layer.send((session.remote_mac, header))

    def send(self, data: tuple[int, int, int, bytes]) -> TransportSession:
        """
        Args:
            data: (src_port:int, dst_mac:int, dst_port:int, payload:bytes)
        """
        src_port, dst_mac, dst_port, payload = data
        src_mac = self.lower_layer.mac_addr
        session = self._get_session(src_mac, src_port, dst_mac, dst_port)

        session.send_buffer += payload
        self.debug_log(
            f"APP SEND_BUFFER session={session} add_len={len(payload)} "
            f"buf_len={len(session.send_buffer)}"
        )

        self._flush_send_buffer(session)
        return session

    def _flush_send_buffer(self, session: TransportSession):
        """
        核心发送逻辑：检查窗口并分片发送数据
        """
        while (
            len(session.send_buffer) > 0
            and len(session.unacked_packets) < session.tx_window_size
        ):
            chunk = session.send_buffer[:MSS]
            session.send_buffer = session.send_buffer[MSS:]

            encapsulated = self.Encapsulate(session=session, data=chunk)
            if encapsulated and self.lower_layer:
                self.lower_layer.send(encapsulated)

        if self.debug_mode:
            self.debug_log(
                f"FLUSH_DONE session={session} "
                f"send_buffer_len={len(session.send_buffer)} "
                f"unacked={list(session.unacked_packets.keys())}"
            )

    def update(self, tick):
        """
        3. Timeout Retransmission: 检查超时并重传
        """
        super().update(tick)

        for _, sessions in self.sessions.items():
            if not sessions.unacked_packets:
                continue
            _, (base_tick, _) = next(iter(sessions.unacked_packets.items()))
            if tick - base_tick > TIMEOUT_TICKS:
                for seq, (sent_tick, packet) in sessions.unacked_packets.items():
                    self.debug_log(
                        f"TIMEOUT retransmitting seq={seq} "
                        f"to {sessions.remote_mac}:{sessions.remote_port}"
                    )
                    if self.lower_layer:
                        self.lower_layer.send((sessions.remote_mac, packet))
                    sessions.unacked_packets[seq] = (tick, packet)
                break


class TransportLayer_SR(ProtocolLayer):
    def __init__(
        self,
        lower_layer: ProtocolLayer = None,
        simulator: PhySimulationEngine = PhySimulationEngine(),
        name: str = "transport_layer",
    ):
        super().__init__(lower_layer=lower_layer, simulator=simulator, name=name)
        # 核心：使用 MAC 地址映射会话状态
        self.sessions: Dict[str, TransportSession] = {}

    def _get_session(self, mac_addr) -> TransportSession:
        if mac_addr not in self.sessions:
            self.sessions[mac_addr] = TransportSession(mac_addr)
        return self.sessions[mac_addr]

    def Encapsulate(self, data: tuple) -> Optional[tuple]:
        """
        封装逻辑：添加 Seq/Ack 头，并缓存用于重传
        Args:
            data: (dst_mac, payload_bytes)
        """
        dst_mac, payload = data
        session = self._get_session(dst_mac)

        # 2. Sequence Numbers: 构建头部 Type=DATA, Seq=Current, Ack=Expected
        header = struct.pack(HEADER_FMT, TYPE_DATA, session.seq_out, session.seq_in)
        packet = header + payload

        # 3. Timeout Retransmission: 记录包和发送时间
        session.unacked_packets[session.seq_out] = (self.simulator.current_tick, packet)

        session.seq_out += 1

        # 返回给下层的数据保持 (dst_mac, packet) 格式
        return (dst_mac, packet)

    def Dencapsulate(self, data: bytes | tuple) -> Optional[bytes]:
        """
        解封装逻辑：处理 ACK，校验 Seq
        Args:
            data: (src_mac, dst_mac, frame_bytes) 来自 MAC 层
        """
        if not isinstance(data, tuple) or len(data) != 3:
            return None

        src_mac, dst_mac, frame = data
        if len(frame) < HEADER_SIZE:
            return None

        # 解析头部
        msg_type = struct.unpack("!B", frame[:1])[0]
        if msg_type == TYPE_DATA:
            msg_type, seq, ack = struct.unpack(HEADER_FMT, frame[:HEADER_SIZE])
            payload: bytes = frame[HEADER_SIZE:]
        elif msg_type == TYPE_ACK:
            msg_type, seq, ack, rwnd = struct.unpack(
                HEAD_FMT_ACK, frame[:ACK_HEADER_SIZE]
            )
            payload: bytes = frame[ACK_HEADER_SIZE:]
        else:
            return None

        session = self._get_session(src_mac)

        # 1. Reliable Transport (ACK处理)
        # 对方发来的 Ack 确认了 seq=Ack

        if msg_type == TYPE_ACK:
            if session.unacked_packets.get(ack):
                del session.unacked_packets[ack]
            session.tx_window_size = max(min(WINDOW_SIZE, rwnd), 1)
            self._flush_send_buffer(session)
            return None  # 纯 ACK 包不向上传递

        elif msg_type == TYPE_DATA:
            self._send_ack(src_mac, seq)
            # 接收数据逻辑
            if seq == session.seq_in:
                # 顺序正确
                combined_data = payload
                session.seq_in += 1
                while (
                    session.recieve_buffer
                    and session.recieve_buffer[0][0] == session.seq_in
                ):
                    _, payload_buf = session.recieve_buffer.pop(0)
                    session.seq_in += 1
                    combined_data += payload_buf
                if not session.recieve_buffer:
                    self._send_ack(src_mac, session.seq_in - 1)
                    pass
                return combined_data
            elif seq < session.seq_in:
                return None
            else:
                if seq >= session.rx_window_size + session.seq_in:
                    return None
                for cache in session.recieve_buffer:
                    if cache[0] == seq:
                        return None
                session.recieve_buffer.append((seq, payload))
                session.recieve_buffer.sort(key=lambda x: x[0])
                return None
                pass

    def _send_ack(self, dst_mac, ack_num):
        """发送纯 ACK 包"""
        session = self._get_session(dst_mac)
        header = struct.pack(
            HEAD_FMT_ACK,
            TYPE_ACK,
            0,
            ack_num,
            WINDOW_SIZE - len(session.recieve_buffer),
        )
        if self.lower_layer:
            self.lower_layer.send((dst_mac, header))

    def handle_data_recieved(self, data):
        payload = self.Dencapsulate(data)
        if payload:
            if self.upper_layer:
                self.upper_layer.handle_data_recieved(payload)
            else:
                self.rx_queue.append(payload)

    def send(self, data: tuple):
        """
        Args:
            data: (dst_mac:str, payload:bytes)
        """
        dst_mac, payload = data
        session = self._get_session(dst_mac)

        #将数据写入发送缓冲区，而不是直接发送
        session.send_buffer += payload

        #尝试刷新缓冲区（分片发送）
        self._flush_send_buffer(session)

    def _flush_send_buffer(self, session: TransportSession):
        """
        检查窗口并分片发送数据
        """
        while (
            len(session.send_buffer) > 0
            and len(session.unacked_packets) < session.tx_window_size
        ):
            # 1. 分片 (Segmentation)
            chunk = session.send_buffer[:MSS]
            session.send_buffer = session.send_buffer[MSS:]

            # 2. 封装并发送
            # 注意：Encapsulate 内部会处理 seq_out++ 和 unacked_packets 的记录
            encapsulated = self.Encapsulate((session.remote_mac, chunk))
            if encapsulated and self.lower_layer:
                self.lower_layer.send(encapsulated)

    def update(self, tick):
        """
        Timeout Retransmission: 检查超时并重传
        """
        super().update(tick)

        for mac, sessions in self.sessions.items():
            """
            遍历会话检查base_seq是否超时
            """
            if not sessions.unacked_packets:
                continue

            for seq, (tick_, packet) in sessions.unacked_packets.items():
                if tick - tick_ > TIMEOUT_TICKS:
                    print(f"[{self.name}] Timeout retransmitting seq {seq} to {mac}")
                    if self.lower_layer:
                        self.lower_layer.send((mac, packet))
                    # 更新发送时间，避免立即再次重传
                    sessions.unacked_packets[seq] = (tick, packet)
