import sys
sys.path.append("..")
from phy import TwistedPair, Cable
from mac import Switcher
from core import PhySimulationEngine
from app.server import HttpServer
from app.client import HttpClient


def test_http_communication():
    """测试 HTTP 客户端和服务器通信"""
    simulator = PhySimulationEngine(time_step_us=1)

    server = HttpServer(simulator=simulator, mac_addr=3, name='server', port=80)
    client = HttpClient(simulator=simulator, mac_addr=2, name='client')
    
    switcher = Switcher(simulator=simulator, mac_addr=0, port_num=2, name='switcher')

    cable = Cable(length=100, attenuation=3, noise_level=3, debug_mode=False)
    tp1 = TwistedPair(cable=cable, simulator=simulator, ID=0)
    tp2 = TwistedPair(cable=cable, simulator=simulator, ID=1)

    server.connect_to(tp1)
    switcher.connect_to(port=0, twisted_pair=tp1)
    client.connect_to(tp2)
    switcher.connect_to(port=1, twisted_pair=tp2)

    server.add_route('/api/data', lambda req: b'{"data": "test_value", "count": 42}')

    responses = []

    def on_response(resp):
        responses.append(resp)
        if resp:
            print(f"Callback with {len(resp['raw'])} byte: status={resp['status_code']}, body={resp['body'][:100]}")

    req1 = client.get(dst_mac=3, dst_port=80, path='/', callback=on_response)
    req2 = client.get(dst_mac=3, dst_port=80, path='/hello', callback=on_response)
    req3 = client.get(dst_mac=3, dst_port=80, path='/api/data', callback=on_response)
    req4 = client.get(dst_mac=3, dst_port=80, path='/not_exist_?', callback=on_response)

    simulator.run(duration_ticks=12000)

    print("\n=== Test Results ===")
    for i, resp in enumerate(responses):
        if resp:
            print(f"Response {i+1}: {resp['raw']}")

    assert len(responses) == 4, f"Should receive 4 responses but got {len(responses)}"
    
    print("\n=== HTTP Test Passed ===")


def test_http_post():
    """测试 POST 请求"""
    simulator = PhySimulationEngine(time_step_us=1)

    server = HttpServer(simulator=simulator, mac_addr=1, name='server', port=80)
    client = HttpClient(simulator=simulator, mac_addr=2, name='client')
    switcher = Switcher(simulator=simulator, mac_addr=0, port_num=2, name='switcher')

    cable = Cable(length=100, attenuation=3, noise_level=4, debug_mode=False)
    tp1 = TwistedPair(cable=cable, simulator=simulator, ID=0)
    tp2 = TwistedPair(cable=cable, simulator=simulator, ID=1)

    server.connect_to(tp1)
    switcher.connect_to(port=0, twisted_pair=tp1)
    client.connect_to(tp2)
    switcher.connect_to(port=1, twisted_pair=tp2)

    # add POST route
    def handle_post(req):
        body = req.get('body', '')
        return f'{{"received": "{body}", "status": "processed"}}'.encode()
    
    server.add_route('/submit', handle_post)

    response_received = []
    
    def on_post_response(resp):
        response_received.append(resp)
        print(f"POST Response: {resp}")

    client.post(
        dst_mac=1, dst_port=80, path='/submit',
        body=b'username=test&password=123',
        callback=on_post_response
    )

    simulator.run(duration_ticks=10000)

    print("\n=== POST Test Results ===")
    for resp in response_received:
        if resp:
            print(f"Status: {resp['status_code']}, Body: {resp['body']}")

    print("\n=== HTTP POST Test Passed ===")


if __name__ == "__main__":
    test_http_communication()
    #test_http_post()