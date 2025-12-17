# test/test_socket.py
import sys
sys.path.append("..")  # 添加父目录到路径
from phy import PhyLayer, TwistedPair, Cable
from mac import MacLayer, Switcher
from tcp import TransportLayer_GBN, TransportLayer_SR, socket
from core import PhySimulationEngine, SimulationEntity

class TestNode(SimulationEntity):
    def __init__(self, simulator:PhySimulationEngine, mac_addr:str, name:str='zero', mode:str='node'):
        super().__init__(name=name)
        self.phy_layer= PhyLayer(lower_layer=None, coding=True, simulator=simulator, name=name)
        self.mac_layer= MacLayer(
                    lower_layer=self.phy_layer, 
                    simulator=simulator,mode=mode, 
                    mac_addr=mac_addr, name=name
                )
        self.tcp_layer= TransportLayer_GBN(lower_layer=self.mac_layer, simulator=simulator, name=name)
        self.socket_layer= self.tcp_layer
        self.socket= socket(tcp_layer=self.tcp_layer)
        self.socket.bind(8080)
        self.socket.setmode('debug')
        self.mac_addr= mac_addr
        self.name= name
        simulator.register_entity(self)

    def connect_to(self, twisted_pair:TwistedPair):
        twisted_pair.connect(
            tx_interface=self.phy_layer.tx_entity,
            rx_interface=self.phy_layer.rx_entity
        )
        print(f'[{self.name}] connected to {twisted_pair}')
    
    def update(self, tick):
        super().update(tick)

def test_tcp_reliable_transfer():
    simulator = PhySimulationEngine(time_step_us=1)

    # 两个节点
    node1 = TestNode(simulator=simulator, mac_addr=1, name='node1')
    node2 = TestNode(simulator=simulator, mac_addr=2, name='node2')

    node1.tcp_layer.setmode('debug')
    node2.tcp_layer.setmode('debug')

    # 信道与连接
    cable = Cable(
        length=100,
        attenuation=4,
        noise_level=4.5,
        debug_mode=False,
    )
    print(f"\n{cable}")

    tp1 = TwistedPair(cable=cable, simulator=simulator, ID=0)

    node1.connect_to(tp1)
    node2.connect_to(tp1)

    # node2: 作为服务器，在 8080 端口监听
    server_sock = node2.socket
    server_sock.listen(num=5)
    server_sock.setmode('debug')

    # node1: 作为客户端，连接到 node2:8080
    client_sock = node1.socket
    client_sock.bind(10000)
    client_sock.setmode('debug')
    client_sock.connect(dst_mac=node2.mac_addr, dst_port=8080)

    # 发送数据
    msg = b'hello, this is a test message for TCP reliable transfer.'*2
    print("\nclient send:", msg, '\n')
    client_sock.send(msg)

    # 运行一段时间，让报文到达 node2
    simulator.run(duration_ticks=10000)

    # 服务器侧：accept，拿到连接 socket
    conn = server_sock.accept()
    assert conn is not None, "accept() should return a connection socket"

    # 服务器侧：recv 数据
    data = conn.recv()
    print("\nserver recv:", data, '\n')
    assert data == msg, f"server should receive {msg}, but got {data}"

def test_two_client_sockets_to_one_server():
    simulator = PhySimulationEngine(time_step_us=1)

    # 三个节点，其中 node2 做服务器
    node1 = TestNode(simulator=simulator, mac_addr=1, name='node1')
    node2 = TestNode(simulator=simulator, mac_addr=2, name='node2')
    node3 = TestNode(simulator=simulator, mac_addr=3, name='node3')
    switcher = Switcher(simulator=simulator, mac_addr=0, port_num=3, name='switcher')

    # 信道与连接
    cable = Cable(
        length=100,
        attenuation=3,
        noise_level=4,
        debug_mode=False,
    )
    print(f"\n{cable}")
    tp1 = TwistedPair(cable=cable, simulator=simulator, ID=0)
    tp2 = TwistedPair(cable=cable, simulator=simulator, ID=1)
    tp3 = TwistedPair(cable=cable, simulator=simulator, ID=2)

    node1.connect_to(tp1)
    switcher.connect_to(port=0, twisted_pair=tp1)
    node2.connect_to(tp2)
    switcher.connect_to(port=1, twisted_pair=tp2)
    node3.connect_to(tp3)
    switcher.connect_to(port=2, twisted_pair=tp3)

    # node2: 作为服务器，在 8080 端口监听
    server_listen = node2.socket
    server_listen.listen(num=5)

    # node1: 两个客户端 socket，端口不同，分别连到 node2:8080
    client_sock1 = socket(tcp_layer=node1.tcp_layer)
    client_sock1.bind(10000)
    client_sock1.setmode('debug')
    client_sock2 = socket(tcp_layer=node1.tcp_layer)
    client_sock2.bind(10001)
    client_sock2.setmode('debug')
    # --- 两个客户端依次发起连接并发送不同的数据 ---
    msg1 = b'hello from client 1'
    msg2 = b'hello from client 2'

    client_sock1.connect(dst_mac=node2.mac_addr, dst_port=8080)
    client_sock1.send(msg1)

    client_sock2.connect(dst_mac=node2.mac_addr, dst_port=8080)
    client_sock2.send(msg2)

    # 运行一段时间，让报文都到达 node2
    simulator.run(duration_ticks=5000)

    # --- 服务器侧：accept 两次，拿到两个不同的连接 socket ---
    conn1 = server_listen.accept()
    conn2 = server_listen.accept()

    print(f'conn1 session: {conn1.session}')
    print(f'conn2 session: {conn2.session}')

    assert conn1 is not None, "first accept() should return a connection socket"
    assert conn2 is not None, "second accept() should return a connection socket"
    assert conn1 is not conn2, "two accepted sockets should be different objects"
    assert conn1.session is not conn2.session, "two connections should have different sessions"

    # --- 从两个连接 socket 分别 recv，检查数据是否对应 ---
    data1 = conn1.recv()
    data2 = conn2.recv()

    print("server conn1 recv:", data1)
    print("server conn2 recv:", data2)

    recv_set = {data1, data2}
    expect_set = {msg1, msg2}
    assert recv_set == expect_set, f"server should receive {expect_set}, but got {recv_set}"

    resp1 = b'response to client 1'
    resp2 = b'response to client 2'
    conn1.send(resp1)
    conn2.send(resp2)

    simulator.run(duration_ticks=5000)

    c1_resp = client_sock1.recv()
    c2_resp = client_sock2.recv()
    print("client1 recv:", c1_resp)
    print("client2 recv:", c2_resp)

    client_sock1.send(b'ack from client 1')
    client_sock2.send(b'ack from client 2')
    simulator.run(duration_ticks=5000)

    print("server conn1 recv after ack:", conn1.recv())
    print("server conn2 recv after ack:", conn2.recv())

    assert c1_resp == resp1
    assert c2_resp == resp2


if __name__ == "__main__":
    test_tcp_reliable_transfer()