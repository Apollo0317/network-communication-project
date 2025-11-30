import sys
sys.path.append("..")
from phy import PhyLayer, TwistedPair, Cable
from mac import MacLayer, Switcher
from tcp import TransportLayer_GBN, socket
from core import PhySimulationEngine, SimulationEntity
from typing import Optional, Dict

class HttpServer(SimulationEntity):
    """
    HTTP server node - non-blocking style
    """
    
    def __init__(self, simulator: PhySimulationEngine, mac_addr: int, 
                 name: str = 'http_server', port: int = 80):
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
        
        # 监听 socket
        self.listen_socket = socket(tcp_layer=self.tcp_layer)
        self.listen_socket.bind(port)
        # self.listen_socket.setmode('debug')
        self.listen_socket.listen(num=10)
        
        # 已建立的连接
        self.connections: list[socket] = []
        
        # 路由表: path -> handler(request) -> response_body
        self.routes: Dict[str, callable] = {}
        self._setup_default_routes()
        
        simulator.register_entity(self)
        print(f"[{name}] HTTP Server started on port {port}")

    def _setup_default_routes(self):
        """设置默认路由"""
        self.routes['/'] = lambda req: b"<html><body><h1>Welcome!</h1></body></html>"
        self.routes['/hello'] = lambda req: b"<html><body><h1>Hello, World!</h1></body></html>"
        self.routes['/status'] = lambda req: b'{"status": "ok"}'

    def add_route(self, path: str, handler: callable):
        """添加自定义路由"""
        self.routes[path] = handler

    def connect_to(self, twisted_pair: TwistedPair):
        twisted_pair.connect(
            tx_interface=self.phy_layer.tx_entity,
            rx_interface=self.phy_layer.rx_entity
        )
        print(f'[{self.name}] connected to {twisted_pair}')

    def _parse_http_request(self, data: bytes) -> Optional[dict]:
        """解析 HTTP 请求"""
        try:
            text = data.decode('utf-8', errors='ignore')
            lines = text.split('\r\n')
            if not lines:
                return None
            
            # 解析请求行: GET /path HTTP/1.1
            request_line = lines[0].split(' ')
            if len(request_line) < 2:
                return None
            
            method = request_line[0]
            path = request_line[1]
            
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
                'method': method,
                'path': path,
                'headers': headers,
                'body': body,
                'raw': data
            }
        except Exception as e:
            print(f"[{self.name}] Parse error: {e}")
            return None

    def _build_http_response(self, status_code: int, body: bytes, 
                              content_type: str = 'text/html') -> bytes:
        """构建 HTTP 响应"""
        status_messages = {
            200: 'OK',
            404: 'Not Found',
            500: 'Internal Server Error'
        }
        status_msg = status_messages.get(status_code, 'Unknown')
        
        response = f"HTTP/1.1 {status_code} {status_msg}\r\n"
        response += f"Content-Type: {content_type}\r\n"
        response += f"Content-Length: {len(body)}\r\n"
        response += "Connection: close\r\n"
        response += "\r\n"
        
        return response.encode('utf-8') + body

    def _handle_request(self, conn: socket, request: dict):
        """处理 HTTP 请求并发送响应"""
        path = request['path']
        method = request['method']
        
        print(f"[{self.name}] {method} {path}")
        
        # 查找路由
        if path in self.routes:
            try:
                body = self.routes[path](request)
                content_type = 'application/json' if path == '/status' else 'text/html'
                response = self._build_http_response(200, body, content_type)
            except Exception as e:
                response = self._build_http_response(500, f"Error: {e}".encode())
        else:
            response = self._build_http_response(404, b"<h1>404 Not Found</h1>")
        
        conn.send(response)
        print(f'[{self.name}] send Response: {response[:50]}...')

    def update(self, tick):
        """非阻塞式更新循环"""
        super().update(tick)
        
        # 1. 尝试 accept 新连接
        new_conn = self.listen_socket.accept()
        if new_conn is not None:
            # new_conn.setmode('debug')
            self.connections.append(new_conn)
            print(f"[{self.name}] New connection accepted at {new_conn.locol_port}")
        
        # 2. 处理已有连接的数据
        for conn in self.connections[:]:  # 用切片避免迭代时修改
            data = conn.recv(4096)
            if data:
                request = self._parse_http_request(data)
                if request:
                    self._handle_request(conn, request)