import sys
import time
import matplotlib.pyplot as plt
from typing import Type
sys.path.append("..")

from phy import PhyLayer, TwistedPair, Cable
from mac import MacLayer, Switcher
from tcp.TransportLayer import TransportLayer_GBN, TransportLayer_SR
from core import PhySimulationEngine, SimulationEntity, ProtocolLayer

# --- 配置参数 ---
DATA_SIZE = 4096 * 8  # 增加数据量: 32KB
TIMEOUT = 500000      # 增加超时时间: 50万 ticks
WINDOW_SIZE = 10      # 增大窗口，让 SR 优势更明显
MSS = 512             # 分片大小

class PerformanceNode(SimulationEntity):
    def __init__(self, simulator: PhySimulationEngine, mac_addr: int, transport_class: Type[ProtocolLayer], name: str):
        super().__init__(name=name)
        self.phy = PhyLayer(lower_layer=None, coding=True, simulator=simulator, name=f"{name}_phy")
        self.mac = MacLayer(lower_layer=self.phy, simulator=simulator, mac_addr=mac_addr, name=f"{name}_mac")
        
        # 动态注入传输层实现类
        self.tcp = transport_class(lower_layer=self.mac, simulator=simulator, name=f"{name}_tcp")
        # 强制统一参数
        self.tcp.WINDOW_SIZE = WINDOW_SIZE 
        self.tcp.MSS = MSS
        
        self.socket = self.tcp
        simulator.register_entity(self)
        self.received_buffer = b''

    def connect(self, tp: TwistedPair):
        tp.connect(self.phy.tx_entity, self.phy.rx_entity)

    def send(self, dst: int, data: bytes):
        self.socket.send((dst, data))

    def fetch_new_data(self):
        """
        非阻塞地从传输层获取新到达的数据
        直接访问 rx_queue 以确保兼容性
        """
        new_bytes = 0
        # 只要队列里有数据，就一直取出来
        while self.socket.rx_queue:
            chunk = self.socket.rx_queue.popleft()
            self.received_buffer += chunk
            new_bytes += len(chunk)
        return len(self.received_buffer)

def run_simulation(transport_class, noise_level, data_payload):
    """运行单次模拟，返回完成所需的 ticks"""
    sim = PhySimulationEngine(time_step_us=1)
    
    # 拓扑：Node1 <---> Cable <---> Node2
    cable = Cable(length=50, attenuation=2, noise_level=noise_level, debug_mode=False)
    tp = TwistedPair(cable=cable, simulator=sim)
    
    sender = PerformanceNode(sim, 1, transport_class, "Sender")
    receiver = PerformanceNode(sim, 2, transport_class, "Receiver")
    
    sender.connect(tp)
    receiver.connect(tp)
    
    # 开始发送
    sender.send(2, data_payload)
    
    start_tick = sim.current_tick
    finished_tick = -1
    target_len = len(data_payload)
    
    # 分步运行
    step = 500 # 每次运行 500 ticks
    
    # 简单的进度条显示
    sys.stdout.write(f"  Running {transport_class.__name__} (Noise={noise_level}): [")
    sys.stdout.flush()
    
    progress_step = TIMEOUT // 20 # 进度条每 5% 更新一次
    
    for i in range(0, TIMEOUT, step):
        sim.run(duration_ticks=step)
        
        # 检查接收进度
        current_len = receiver.fetch_new_data()
        
        if current_len >= target_len:
            finished_tick = sim.current_tick
            break
            
        # 更新进度条
        if i % progress_step < step:
            sys.stdout.write(".")
            sys.stdout.flush()
            
    sys.stdout.write("] ")
    
    if finished_tick == -1:
        print(f"TIMEOUT! ({current_len}/{target_len} bytes)")
        return float('inf')
    
    duration = finished_tick - start_tick
    print(f"Done in {duration} ticks")
    return duration

def main():
    # 噪声等级: 0=无误码, 4=低误码, 8=中误码, 12=高误码
    noise_levels = [0, 4, 8, 12] 
    payload = b'X' * DATA_SIZE
    
    results_gbn = []
    results_sr = []
    
    print(f"=== Performance Comparison ===")
    print(f"Data Size: {DATA_SIZE} bytes")
    print(f"Window Size: {WINDOW_SIZE}")
    print(f"Timeout Limit: {TIMEOUT} ticks")
    print("=" * 60)
    
    for noise in noise_levels:
        # 运行 GBN
        ticks_gbn = run_simulation(TransportLayer_GBN, noise, payload)
        results_gbn.append(ticks_gbn)
        
        # 运行 SR
        ticks_sr = run_simulation(TransportLayer_SR, noise, payload)
        results_sr.append(ticks_sr)

    # 打印汇总表
    print("\n" + "=" * 60)
    print(f"{'Noise':<10} | {'GBN Ticks':<15} | {'SR Ticks':<15} | {'Improvement':<15}")
    print("-" * 60)
    
    for i, noise in enumerate(noise_levels):
        t_gbn = results_gbn[i]
        t_sr = results_sr[i]
        
        if t_gbn == float('inf'):
            imp = "N/A"
            t_gbn_str = "TIMEOUT"
        else:
            t_gbn_str = str(t_gbn)
            if t_sr != float('inf'):
                imp = f"{((t_gbn - t_sr) / t_gbn * 100):.1f}%"
            else:
                imp = "-inf%"

        t_sr_str = "TIMEOUT" if t_sr == float('inf') else str(t_sr)
            
        print(f"{noise:<10} | {t_gbn_str:<15} | {t_sr_str:<15} | {imp:<15}")

    # 绘图
    try:
        plt.figure(figsize=(10, 6))
        # 过滤掉 INF 数据以便绘图
        valid_indices = [i for i, t in enumerate(results_gbn) if t != float('inf') and results_sr[i] != float('inf')]
        
        if valid_indices:
            valid_noise = [noise_levels[i] for i in valid_indices]
            valid_gbn = [results_gbn[i] for i in valid_indices]
            valid_sr = [results_sr[i] for i in valid_indices]
            
            plt.plot(valid_noise, valid_gbn, 'o-', label='GBN', color='red', linewidth=2)
            plt.plot(valid_noise, valid_sr, 's-', label='SR', color='blue', linewidth=2)
            
            plt.xlabel('Noise Level (Higher = More Errors)')
            plt.ylabel('Ticks to Finish (Lower is Better)')
            plt.title(f'GBN vs SR Performance ({DATA_SIZE} bytes)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.savefig('protocol_comparison.png')
            print("\nChart saved to protocol_comparison.png")
        else:
            print("\nNot enough valid data points to plot chart.")
            
    except ImportError:
        print("\nMatplotlib not found, skipping chart generation.")

if __name__ == "__main__":
    main()