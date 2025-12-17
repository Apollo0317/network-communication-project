import sys
sys.path.append("..")
from phy import PhyLayer, TwistedPair, Cable
from mac import MacLayer
from tcp import TransportLayer_GBN, socket
from core import PhySimulationEngine, SimulationEntity
from typing import Optional, Callable
from enum import Enum

class RequestState(Enum):
    IDLE = 0
    PENDING = 1
    COMPLETED = 2

class HttpRequest:
    """HTTP 请求封装"""
    def __init__(self, method: str, path: str, headers: dict = None, body: bytes = b''):
        self.method = method
        self.path = path
        self.headers = headers or {}
        self.body = body
        self.state = RequestState.IDLE
        self.response: Optional[dict] = None
        self.callback: Optional[Callable] = None
        self.sock: Optional[socket] = None

class HttpClient(SimulationEntity):
    """
    HTTP client node - non-blocking style
    """
    
    def __init__(self, simulator: PhySimulationEngine, mac_addr: int, name: str = 'http_client'):
        super().__init__(name=name)
        self.simulator = simulator
        self.mac_addr = mac_addr
        
        # 协议栈
        self.phy_layer = PhyLayer(lower_layer=None, coding=True, simulator=simulator, name=name)
        self.mac_layer = MacLayer(
            lower_layer=self.phy_layer,
            simulator=simulator, mode='node',
            mac_addr=mac_addr, name=name
        )
        self.tcp_layer = TransportLayer_GBN(lower_layer=self.mac_layer, simulator=simulator, name=name)
        
        # 端口分配
        self._next_port = 10000
        
        # 待处理的请求队列
        self.pending_requests: list[HttpRequest] = []
        
        simulator.register_entity(self)
        print(f"[{name}] HTTP Client initialized")

    def connect_to(self, twisted_pair: TwistedPair):
        twisted_pair.connect(
            tx_interface=self.phy_layer.tx_entity,
            rx_interface=self.phy_layer.rx_entity
        )
        print(f'[{self.name}] connected to {twisted_pair}')

    def _allocate_port(self) -> int:
        port = self._next_port
        self._next_port += 1
        return port

    def _build_http_request(self, method: str, path: str, host: str,
                             headers: dict = None, body: bytes = b'') -> bytes:
        """Build HTTP request bytes"""
        request = f"{method} {path} HTTP/1.1\r\n"
        request += f"Host: {host}\r\n"
        
        if headers:
            for key, value in headers.items():
                request += f"{key}: {value}\r\n"
        
        if body:
            request += f"Content-Length: {len(body)}\r\n"
        
        request += "Connection: close\r\n"
        request += "\r\n"
        
        return request.encode('utf-8') + body

    def _parse_http_response(self, data: bytes) -> Optional[dict]:
        try:
            text = data.decode('utf-8', errors='ignore')
            lines = text.split('\r\n')
            if not lines:
                return None
            
            # 解析状态行: HTTP/1.1 200 OK
            status_line = lines[0].split(' ', 2)
            if len(status_line) < 2:
                return None
            
            status_code = int(status_line[1])
            status_msg = status_line[2] if len(status_line) > 2 else ''
            
            # 解析 headers
            headers = {}
            body_start = 0
            for i, line in enumerate(lines[1:], 1):
                if line == '':
                    body_start = i + 1
                    break
                if ':' in line:
                    key, value = line.split(':', 1)
                    headers[key.strip()] = value.strip()
            
            body = '\r\n'.join(lines[body_start:]) if body_start < len(lines) else ''
            
            return {
                'status_code': status_code,
                'status_msg': status_msg,
                'headers': headers,
                'body': body,
                'raw': data
            }
        except Exception as e:
            print(f"[{self.name}] Response parse error: {e}")
            return None

    def get(self, dst_mac: int, dst_port: int, path: str, 
            callback: Callable = None) -> HttpRequest:
        """发起 GET 请求(async)"""
        return self._send_request('GET', dst_mac, dst_port, path, callback=callback)

    def post(self, dst_mac: int, dst_port: int, path: str, 
             body: bytes = b'', callback: Callable = None) -> HttpRequest:
        """发起 POST 请求（async）"""
        return self._send_request('POST', dst_mac, dst_port, path, body=body, callback=callback)

    def _send_request(self, method: str, dst_mac: int, dst_port: int, 
                      path: str, body: bytes = b'', callback: Callable = None) -> HttpRequest:
        """发送 HTTP 请求"""
        req = HttpRequest(method, path, body=body)
        req.callback = callback
        req.state = RequestState.PENDING
        
        # 创建 socket 并连接
        sock = socket(tcp_layer=self.tcp_layer)
        sock.bind(self._allocate_port())
        # sock.setmode('debug')
        sock.connect(dst_mac=dst_mac, dst_port=dst_port)
        req.sock = sock
        
        # 构建并发送 HTTP 请求
        http_data = self._build_http_request(method, path, f"{dst_mac}:{dst_port}", body=body)
        sock.send(http_data)
        
        self.pending_requests.append(req)
        print(f"[TICK {self.current_tick}][{self.name}] {method} request sent to {dst_mac}:{dst_port}{path}")
        
        return req

    def update(self, tick):
        super().update(tick)
        
        for req in self.pending_requests[:]:
            if req.state != RequestState.PENDING:
                continue
            
            data = req.sock.recv(length=4096)
            if data:
                response = self._parse_http_response(data)
                req.response = response
                req.state = RequestState.COMPLETED
                
                if response:
                    print(f"[TICK {self.current_tick}][{self.name}] Response received: {response['raw']}\n")
                
                # 调用回调
                if req.callback:
                    req.callback(response)
                
                self.pending_requests.remove(req)