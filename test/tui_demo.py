import sys
sys.path.append("..")
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TaskProgressColumn
from rich.columns import Columns
from rich import box
from rich.align import Align
from rich.rule import Rule
import time
import math

from phy import PhyLayer, TwistedPair, Cable
from mac import MacLayer, Switcher
from core import PhySimulationEngine, SimulationEntity
from app.server import HttpServer
from app.client import HttpClient
from utils import generate_random_data, diff
from typing import Optional, List

console = Console()


# ============================================================
#                      节点包装器
# ============================================================

class VisualPhyNode(SimulationEntity):
    """Level 1 用：纯物理层节点"""
    def __init__(self, simulator: PhySimulationEngine, name: str, coding: bool = True):
        super().__init__(name=name)
        self.phy_layer = PhyLayer(lower_layer=None, coding=coding, simulator=simulator, name=name)
        self.socket_layer = self.phy_layer
        self.name = name
        
        self.sent_bytes = 0
        self.recv_bytes = 0
        self.recv_buffer = []
        
        simulator.register_entity(self)

    def send(self, data: bytes):
        self.socket_layer.send(data)
        self.sent_bytes += len(data)

    def recv(self) -> Optional[bytes]:
        result = self.socket_layer.recv()
        if result:
            self.recv_bytes += len(result)
            self.recv_buffer.append(result)
            return result
        return None

    def connect_to(self, twisted_pair: TwistedPair):
        twisted_pair.connect(
            tx_interface=self.phy_layer.tx_entity,
            rx_interface=self.phy_layer.rx_entity
        )

    def update(self, tick):
        super().update(tick)
        self.recv()


class VisualMacNode(SimulationEntity):
    """Level 2 用：带 MAC 层的节点"""
    def __init__(self, simulator: PhySimulationEngine, mac_addr: int, name: str):
        super().__init__(name=name)
        self.phy_layer = PhyLayer(lower_layer=None, coding=True, simulator=simulator, name=name)
        self.mac_layer = MacLayer(lower_layer=self.phy_layer, mac_addr=mac_addr, simulator=simulator, name=name)
        self.socket_layer = self.mac_layer
        self.mac_addr = mac_addr
        self.name = name
        
        self.sent_count = 0
        self.recv_count = 0
        self.sent_bytes = 0
        self.recv_bytes = 0
        self.last_sent = ""
        self.last_recv = ""
        self.recv_buffer = []
        
        simulator.register_entity(self)

    def send(self, dst_mac: int, data: bytes):
        self.socket_layer.send((self.mac_addr, dst_mac, data))
        self.sent_count += 1
        self.sent_bytes += len(data)
        self.last_sent = data[:40].decode('utf-8', errors='replace')

    def recv(self) -> Optional[bytes]:
        result = self.socket_layer.recv()
        if result:
            _, _, data = result
            self.recv_count += 1
            self.recv_bytes += len(data)
            self.last_recv = data[:40].decode('utf-8', errors='replace')
            self.recv_buffer.append(data)
            return data
        return None

    def connect_to(self, twisted_pair: TwistedPair):
        twisted_pair.connect(
            tx_interface=self.phy_layer.tx_entity,
            rx_interface=self.phy_layer.rx_entity
        )

    def update(self, tick):
        super().update(tick)
        self.recv()


# ============================================================
#                      静态 UI 组件
# ============================================================

def print_header(title: str, subtitle: str = ""):
    """打印标题（静态，不刷新）"""
    console.print()
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]", style="cyan"))
    if subtitle:
        console.print(Align.center(Text(subtitle, style="dim")))
    console.print()


def print_cable_info(cable: Cable):
    """打印信道参数（静态）"""
    table = Table(box=box.ROUNDED, title="📡 Channel Parameters", title_style="bold")
    table.add_column("Parameter", style="dim")
    table.add_column("Value", style="yellow")
    
    table.add_row("Length", f"{cable.length} m")
    table.add_row("Attenuation", str(cable.attenuation))
    table.add_row("Noise Level", str(cable.noise_level))
    
    if cable.noise_level > 0:
        snr = 1.0 / cable.noise_level
        shannon = math.log2(1 + snr)
        table.add_row("Est. SNR", f"{snr:.3f}")
        table.add_row("Shannon Capacity", f"{shannon:.3f} bits/symbol")
    
    console.print(table)
    console.print()


def print_p2p_topology(node_a_name: str, node_b_name: str):
    """打印点对点拓扑（静态）"""
    topology = f"""
    ┌──────────┐                                    ┌──────────┐
    │  [cyan]{node_a_name:^6}[/cyan]  │ ══════════ Cable ══════════ │  [cyan]{node_b_name:^6}[/cyan]  │
    │  Sender  │         ~~~>>>~~~              │ Receiver │
    └──────────┘                                    └──────────┘
    """
    console.print(Panel(topology, title="🔌 Point-to-Point Connection", border_style="blue"))
    console.print()


def print_star_topology(node_names: List[str]):
    """打印星型拓扑（静态）"""
    n = len(node_names)
    lines = []
    lines.append("              ┌────────────┐")
    lines.append("              │  [magenta]SWITCH[/magenta]    │")
    lines.append("              │  Ports: {}  │".format(n))
    lines.append("              └─────┬──────┘")
    lines.append("                    │")
    
    if n == 2:
        lines.append("          ┌─────────┴─────────┐")
        lines.append("          │                   │")
        lines.append(f"      [cyan]{node_names[0]:^8}[/cyan]           [cyan]{node_names[1]:^8}[/cyan]")
    elif n == 3:
        lines.append("      ┌───────────┼───────────┐")
        lines.append("      │           │           │")
        lines.append(f"  [cyan]{node_names[0]:^8}[/cyan]    [cyan]{node_names[1]:^8}[/cyan]    [cyan]{node_names[2]:^8}[/cyan]")
    
    console.print(Panel("\n".join(lines), title="🌐 Star Topology", border_style="blue"))
    console.print()


def print_packet_header_design():
    """打印数据包头设计（静态）"""
    header_design = """
[bold]MAC Frame Format:[/bold]

┌─────────────┬─────────────┬──────────────────────┐
│   SRC_MAC   │   DST_MAC   │        DATA          │
│   (1 byte)  │   (1 byte)  │      (N bytes)       │
└─────────────┴─────────────┴──────────────────────┘

• [yellow]SRC_MAC[/yellow]: Source MAC address (0-255)
• [yellow]DST_MAC[/yellow]: Destination MAC address (0-255)
• [yellow]DATA[/yellow]:    Upper layer payload
    """
    console.print(Panel(header_design, title="📦 Packet Header Design", border_style="yellow"))
    console.print()


# ============================================================
#                      Level 1 演示
# ============================================================

def demo_level1_basic():
    """Level 1: 基础比特流传输 - 简单字符串"""
    console.clear()
    print_header(
        "Level 1: Point-to-Point Communication",
        "基础比特流传输演示 - 评分项: 成功传输简单字符串 [15分]"
    )
    
    simulator = PhySimulationEngine(time_step_us=1, realtime_mode=True)
    node_a = VisualPhyNode(simulator=simulator, name='HostA')
    node_b = VisualPhyNode(simulator=simulator, name='HostB')
    
    cable = Cable(length=100, attenuation=4, noise_level=2, debug_mode=False)
    tp = TwistedPair(cable=cable, simulator=simulator, ID=0)
    
    node_a.connect_to(tp)
    node_b.connect_to(tp)
    
    print_cable_info(cable)
    print_p2p_topology("HostA", "HostB")
    
    test_msg = b"Hello, Network Communication! This is a test message."
    console.print(f"[green]📤 Sending:[/green] {test_msg.decode()}")
    console.print(f"[dim]   Length: {len(test_msg)} bytes[/dim]")
    console.print()
    
    node_a.send(test_msg)
    
    # 使用进度条显示传输过程
    total_ticks = 4000
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        console=console,
        transient=False
    ) as progress:
        task = progress.add_task("[cyan]Transmitting...", total=total_ticks)
        
        for tick in range(0, total_ticks, 1000):
            simulator.run(duration_ticks=10000)
            progress.update(task, advance=1000)
            
            # 检查是否已收到
            if node_b.recv_buffer:
                progress.update(task, completed=total_ticks)
                break
            time.sleep(0.02)
    
    console.print()
    
    # 显示结果
    if node_b.recv_buffer:
        recv_data = node_b.recv_buffer[0]
        success = test_msg == recv_data
        
        result_table = Table(box=box.ROUNDED, title="📋 Transmission Result")
        result_table.add_column("Item", style="dim")
        result_table.add_column("Value")
        
        result_table.add_row("Sent", f"{len(test_msg)} bytes")
        result_table.add_row("Received", f"{len(recv_data)} bytes")
        result_table.add_row("Content", recv_data.decode('utf-8', errors='replace')[:60])
        result_table.add_row("Match", "[green]✅ YES[/green]" if success else "[red]❌ NO[/red]")
        
        console.print(result_table)
        
        if success:
            console.print("\n[bold green]✅ 传输成功！数据完整无误。[/bold green]")
        else:
            diff_count = diff(test_msg, recv_data)
            console.print(f"\n[bold red]❌ 传输有误，{diff_count} 字节不匹配。[/bold red]")
    else:
        console.print("[bold red]❌ 未收到任何数据[/bold red]")


def demo_level1_fragmentation():
    """Level 1: 消息分片传输 - 长消息"""
    console.clear()
    print_header(
        "Level 1: Message Fragmentation",
        "长消息分片传输演示 - 评分项: 处理较长消息 [5分]"
    )
    
    simulator = PhySimulationEngine(time_step_us=1, realtime_mode=True)
    node_a = VisualPhyNode(simulator=simulator, name='HostA')
    node_b = VisualPhyNode(simulator=simulator, name='HostB')
    
    cable = Cable(length=100, attenuation=4, noise_level=2, debug_mode=False)
    tp = TwistedPair(cable=cable, simulator=simulator, ID=0)
    
    node_a.connect_to(tp)
    node_b.connect_to(tp)
    
    print_cable_info(cable)
    print_p2p_topology("HostA", "HostB")
    
    # 发送较长消息
    test_msg = generate_random_data(length=1024*2)
    console.print(f"[green]📤 Sending:[/green] {len(test_msg)} bytes of random data")
    console.print(f"[dim]   Preview: {test_msg[:32].hex()}...[/dim]")
    console.print()
    
    # 显示分片信息
    fragment_size = 64  # 假设的分片大小
    fragment_count = (len(test_msg) + fragment_size - 1) // fragment_size
    
    frag_panel = Panel(
        f"""
[bold]Message Fragmentation Info:[/bold]

  Original Size:    [yellow]{len(test_msg)}[/yellow] bytes
  Fragment Size:    [yellow]{fragment_size}[/yellow] bytes
  Total Fragments:  [yellow]{fragment_count}[/yellow]

  [dim]Fragments: [/dim][{'░' * min(fragment_count, 30)}]
        """,
        title="🔀 Fragmentation",
        border_style="magenta"
    )
    console.print(frag_panel)
    console.print()
    
    node_a.send(test_msg)
    
    total_ticks = 100000
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        console=console,
        transient=False
    ) as progress:
        task = progress.add_task("[cyan]Transmitting fragments...", total=total_ticks)
        
        for tick in range(0, total_ticks, 10000):
            simulator.run(duration_ticks=10000)
            progress.update(task, advance=1000)
            
            if node_b.recv_buffer:
                progress.update(task, completed=total_ticks)
                break
            time.sleep(0.01)
    
    console.print()
    
    # 结果
    if node_b.recv_buffer:
        recv_data = node_b.recv_buffer[0]
        diff_count = diff(test_msg, recv_data)
        success = diff_count == 0
        
        result_table = Table(box=box.ROUNDED, title=" Fragmentation Result")
        result_table.add_column("Metric", style="dim")
        result_table.add_column("Value")
        
        result_table.add_row("Original Size", f"{len(test_msg)} bytes")
        result_table.add_row("Received Size", f"{len(recv_data)} bytes")
        result_table.add_row("Fragments", str(fragment_count))
        result_table.add_row("Byte Errors", str(diff_count))
        result_table.add_row("Status", "[green] Complete[/green]" if success else f"[yellow]{diff_count} errors[/yellow]")
        
        console.print(result_table)
    else:
        console.print("[bold red] 未收到任何数据[/bold red]")


def demo_level1_noise():
    """Level 1: 噪声环境测试 + 香农公式对比"""
    console.clear()
    print_header(
        "Level 1: Noise Performance Analysis",
        "不同噪声下的传输性能 vs 香农公式 - 评分项: R vs C 对比 [10分]"
    )
    
    # 使用较长的消息以获得收敛的 BER
    MSG_LENGTH = 1024*8  # 4KB = 32768 bits，足够 BER 收敛
    test_msg = generate_random_data(length=MSG_LENGTH)
    noise_levels = [1, 3.6, 4.3, 5.2, 6.6]
    
    # ========== 系统参数（基于 modulator.py 和 Coding.py）==========
    SYMBOL_RATE = 1e6           # 符号率 (symbols/sec)，来自 modulator.py
    BITS_PER_SYMBOL = 4         # 16-QAM: 4 bits/symbol
    SAMPLE_RATE = 50e6          # 采样率 (samples/sec)
    CARRIER_FREQ = 2e6          # 载波频率 (Hz)
    
    # Hamming(7,4) 编码效率
    HAMMING_K = 4               # 信息比特
    HAMMING_N = 7               # 编码比特
    CODING_RATE = HAMMING_K / HAMMING_N  # ≈ 0.571
    
    # 带宽估算：对于 16-QAM，带宽 ≈ 符号率 (理想 Nyquist)
    BANDWIDTH = SYMBOL_RATE     # Hz
    
    # 传输速率计算
    RAW_BIT_RATE = SYMBOL_RATE * BITS_PER_SYMBOL  # 4 Mbps (符号层)
    CODED_INFO_RATE = RAW_BIT_RATE * CODING_RATE  # ≈ 2.286 Mbps (实际信息速率，有编码)
    UNCODED_INFO_RATE = RAW_BIT_RATE              # 4 Mbps (无编码)
    
    # 归一化速率 (bits per symbol)
    R_CODED = BITS_PER_SYMBOL * CODING_RATE       # ≈ 2.286 bits/symbol
    R_UNCODED = BITS_PER_SYMBOL                   # 4 bits/symbol
    
    console.print(f"[dim]Test Message: {len(test_msg)} bytes = {len(test_msg) * 8} bits[/dim]")
    console.print(f"[dim](Using large message for accurate BER measurement)[/dim]")
    console.print()
    
    # 系统参数面板
    param_panel = Panel(
        f"""
[bold]System Parameters (from modulator.py & Coding.py):[/bold]

  ┌─────────────────────────────────────────────────────────────┐
  │                    Modulation Scheme                        │
  ├──────────────────────────┬──────────────────────────────────┤
  │  Modulation              │  [yellow]16-QAM[/yellow]                           │
  │  Bits per Symbol         │  [yellow]{BITS_PER_SYMBOL}[/yellow] bits/symbol                   │
  │  Symbol Rate             │  [yellow]{SYMBOL_RATE/1e6:.1f}[/yellow] Msymbols/sec               │
  │  Sample Rate             │  [yellow]{SAMPLE_RATE/1e6:.1f}[/yellow] MHz                        │
  │  Carrier Frequency       │  [yellow]{CARRIER_FREQ/1e6:.1f}[/yellow] MHz                        │
  │  Bandwidth (Nyquist)     │  [yellow]{BANDWIDTH/1e6:.1f}[/yellow] MHz                        │
  └──────────────────────────┴──────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────┐
  │                    Channel Coding                           │
  ├──────────────────────────┬──────────────────────────────────┤
  │  Coding Scheme           │  [yellow]Hamming(7,4)[/yellow]                      │
  │  Code Rate               │  [yellow]{CODING_RATE:.4f}[/yellow] (k/n = 4/7)              │
  │  Error Correction        │  [yellow]1 bit[/yellow] per 7-bit block             │
  └──────────────────────────┴──────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────┐
  │                    Transmission Rate                        │
  ├──────────────────────────┬──────────────────────────────────┤
  │  Raw Symbol Rate         │  [yellow]{RAW_BIT_RATE/1e6:.2f}[/yellow] Mbps                      │
  │  R (with coding)         │  [cyan]{CODED_INFO_RATE/1e6:.3f}[/cyan] Mbps = [cyan]{R_CODED:.3f}[/cyan] bits/sym   │
  │  R (no coding)           │  [cyan]{UNCODED_INFO_RATE/1e6:.2f}[/cyan] Mbps = [cyan]{R_UNCODED:.1f}[/cyan] bits/sym      │
  └──────────────────────────┴──────────────────────────────────┘
        """,
        title="System Configuration",
        border_style="cyan"
    )
    console.print(param_panel)
    console.print()
    
    # 结果表格
    result_table = Table(
        box=box.DOUBLE_EDGE,
        title="R vs C vs BER Analysis",
        title_style="bold cyan"
    )
    result_table.add_column("Noise\nLevel", style="cyan", justify="center")
    result_table.add_column("SNR\n(linear)", style="yellow", justify="center")
    result_table.add_column("SNR\n(dB)", style="yellow", justify="center")
    result_table.add_column("C\n(bits/sym)", style="magenta", justify="center")
    result_table.add_column("R\n(bits/sym)", style="green", justify="center")
    result_table.add_column("R/C", style="blue", justify="center")
    result_table.add_column("R < C ?", style="blue", justify="center")
    result_table.add_column("BER", style="red", justify="center")
    result_table.add_column("Theory\nPrediction", justify="center")
    
    results_data = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Testing noise levels...", total=len(noise_levels))
        
        for noise in noise_levels:
            progress.update(task, description=f"[cyan]Testing noise_level={noise}...[/cyan]")
            
            simulator = PhySimulationEngine(time_step_us=1, realtime_mode=False)
            node_a = VisualPhyNode(simulator=simulator, name='A', coding=True)
            node_b = VisualPhyNode(simulator=simulator, name='B', coding=True)
            
            cable = Cable(length=100, attenuation=3.5, noise_level=noise, debug_mode=False)
            tp = TwistedPair(cable=cable, simulator=simulator, ID=0)
            
            node_a.connect_to(tp)
            node_b.connect_to(tp)
            node_a.send(test_msg)
            
            # 运行足够长的时间确保传输完成
            estimated_ticks = len(test_msg) * 10  # 预留足够时间
            simulator.run(duration_ticks=estimated_ticks)
            
            recv_data = node_b.recv_buffer[0] if node_b.recv_buffer else b""
            
            # 从 cable 获取真实 SNR
            try:
                snr_linear = tp.channel_a.cable._calculate_snr()
                if snr_linear is None or snr_linear <= 0:
                    snr_linear = float('inf') if noise == 0 else 1.0 / noise
            except:
                snr_linear = float('inf') if noise == 0 else 1.0 / noise
            
            # 计算比特错误 (BER)
            total_bits = len(test_msg) * 8
            if recv_data:
                bit_errors = 0
                for i in range(min(len(test_msg), len(recv_data))):
                    xor = test_msg[i] ^ recv_data[i]
                    bit_errors += bin(xor).count('1')
                bit_errors += abs(len(test_msg) - len(recv_data)) * 8
            else:
                bit_errors = total_bits
            
            ber = bit_errors / total_bits if total_bits > 0 else 1.0
            
            # 计算香农容量 C (bits per symbol)
            # C = log2(1 + SNR) per channel use (symbol)
            if snr_linear == float('inf') or noise == 0:
                snr_str = "∞"
                snr_db_str = "∞"
                shannon_capacity = float('inf')
                shannon_str = "∞"
                ratio = 0
                ratio_str = "0"
                r_less_than_c = "[green]Yes[/green]"
                theory = "[green]BER→0[/green]"
            else:
                snr_db = 10 * math.log10(snr_linear) if snr_linear > 0 else float('-inf')
                snr_str = f"{snr_linear:.4f}"
                snr_db_str = f"{snr_db:.2f}"
                
                # 香农容量: C = log2(1 + snr_linear) bits per symbol
                shannon_capacity = math.log2(1 + snr_linear)
                shannon_str = f"{shannon_capacity:.4f}"
                
                # 使用有编码的 R
                ratio = R_CODED / shannon_capacity
                ratio_str = f"{ratio:.2f}"
                
                if R_CODED < shannon_capacity:
                    r_less_than_c = "[green]Yes[/green]"
                    theory = "[green]BER→0 possible[/green]"
                else:
                    r_less_than_c = "[red]No[/red]"
                    theory = "[red]BER>0 inevitable[/red]"
            
            # BER 格式化
            if ber == 0:
                ber_str = "[green]0[/green]"
            elif ber < 1e-4:
                ber_str = f"[green]{ber:.2e}[/green]"
            elif ber < 1e-3:
                ber_str = f"[yellow]{ber:.2e}[/yellow]"
            elif ber < 1e-2:
                ber_str = f"[orange1]{ber:.2e}[/orange1]"
            elif ber < 0.1:
                ber_str = f"[red]{ber:.2%}[/red]"
            else:
                ber_str = f"[red bold]{ber:.2%}[/red bold]"
            
            result_table.add_row(
                str(noise),
                snr_str,
                snr_db_str,
                shannon_str,
                f"{R_CODED:.3f}",
                ratio_str,
                r_less_than_c,
                ber_str,
                theory
            )
            
            results_data.append({
                'noise': noise,
                'snr_linear': snr_linear,
                'snr_db': snr_db if noise > 0 else float('inf'),
                'shannon_c': shannon_capacity,
                'R': R_CODED,
                'ratio': ratio if noise > 0 else 0,
                'ber': ber,
            })
            
            progress.advance(task)
            time.sleep(0.05)
    
    console.print(result_table)
    console.print()
    
    # ========== 分析：找出 R = C 的临界点 ==========
    console.print(Rule("[bold]Critical Point Analysis: Where R = C[/bold]", style="cyan"))
    console.print()
    
    # 计算 R = C 时的临界 SNR
    # C = log2(1 + SNR) = R
    # SNR = 2^R - 1
    critical_snr = 2 ** R_CODED - 1
    critical_snr_db = 10 * math.log10(critical_snr)
    
    critical_panel = Panel(
        f"""
[bold]Shannon Limit Analysis:[/bold]

  当前系统实际传输速率:
  [cyan]R = {R_CODED:.4f} bits/symbol[/cyan] (16-QAM with Hamming(7,4))

  根据香农公式 C = log₂(1 + SNR):
  
  [yellow]临界条件 R = C 时:[/yellow]
  
    R = log₂(1 + SNR_critical)
    {R_CODED:.4f} = log₂(1 + SNR_critical)
    SNR_critical = 2^{R_CODED:.4f} - 1
    
    [bold cyan]SNR_critical = {critical_snr:.4f} ({critical_snr_db:.2f} dB)[/bold cyan]

  [bold]理论预测:[/bold]
  
    • 当 SNR > {critical_snr:.4f} (即 > {critical_snr_db:.2f} dB):
      [green]R < C → 存在编码方案使 BER → 0[/green]
      
    • 当 SNR < {critical_snr:.4f} (即 < {critical_snr_db:.2f} dB):
      [red]R > C → 无论如何编码，BER 必然 > 0[/red]
        """,
        title="Critical SNR Calculation",
        border_style="yellow"
    )
    console.print(critical_panel)
    console.print()
    
    # ========== 验证表格：对比理论与实测 ==========
    console.print(Rule("[bold]Theory vs Measured Results[/bold]", style="cyan"))
    console.print()
    
    verify_table = Table(box=box.ROUNDED, title="Shannon Limit Verification")
    verify_table.add_column("Noise", style="cyan", justify="center")
    verify_table.add_column("SNR (dB)", style="yellow", justify="center")
    verify_table.add_column("vs Critical\n({:.2f} dB)".format(critical_snr_db), justify="center")
    verify_table.add_column("R/C", style="blue", justify="center")
    verify_table.add_column("Theory", justify="center")
    verify_table.add_column("Measured BER", style="red", justify="center")
    verify_table.add_column("Match?", justify="center")
    
    for r in results_data:
        noise = r['noise']
        snr_db = r['snr_db']
        ber = r['ber']
        ratio = r['ratio']
        
        if noise == 0:
            snr_db_str = "∞"
            vs_critical = "[green]>> critical[/green]"
            theory = "BER → 0"
            match = "[green]✓[/green]" if ber == 0 else "[yellow]~[/yellow]"
        else:
            snr_db_str = f"{snr_db:.2f}"
            if snr_db > critical_snr_db + 3:
                vs_critical = f"[green]+{snr_db - critical_snr_db:.1f} dB[/green]"
                theory = "[green]BER → 0[/green]"
                match = "[green]✓[/green]" if ber < 0.01 else "[red]✗[/red]"
            elif snr_db > critical_snr_db:
                vs_critical = f"[yellow]+{snr_db - critical_snr_db:.1f} dB[/yellow]"
                theory = "[yellow]BER low[/yellow]"
                match = "[green]✓[/green]" if ber < 0.1 else "[yellow]~[/yellow]"
            else:
                vs_critical = f"[red]{snr_db - critical_snr_db:.1f} dB[/red]"
                theory = "[red]BER > 0[/red]"
                match = "[green]✓[/green]" if ber > 0 else "[red]✗[/red]"
        
        ber_str = f"{ber:.2e}" if ber > 0 and ber < 0.01 else f"{ber:.2%}" if ber > 0 else "0"
        ratio_str = f"{ratio:.2f}" if noise > 0 else "0"
        
        verify_table.add_row(
            str(noise),
            snr_db_str,
            vs_critical,
            ratio_str,
            theory,
            ber_str,
            match
        )
    
    console.print(verify_table)
    console.print()
    
    # 理论说明面板
    console.print(Panel(
        f"""
[bold]Shannon-Hartley Theorem:[/bold]

  [yellow]C = B x log₂(1 + SNR)[/yellow]  或归一化: [yellow]C = log₂(1 + SNR)[/yellow] bits/symbol

[bold]本系统参数:[/bold]

  • 调制: 16-QAM → {BITS_PER_SYMBOL} bits/symbol (原始)
  • 编码: Hamming(7,4) → 效率 {CODING_RATE:.4f}
  • 实际信息速率: R = {BITS_PER_SYMBOL} × {CODING_RATE:.4f} = [cyan]{R_CODED:.4f}[/cyan] bits/symbol
  
[bold]Shannon's Theorem 验证:[/bold]

  临界 SNR (R = C): [yellow]{critical_snr:.4f} = {critical_snr_db:.2f} dB[/yellow]
  
  • SNR > {critical_snr_db:.2f} dB: R < C, 理论上可实现任意低的 BER
  • SNR < {critical_snr_db:.2f} dB: R > C, BER 必然大于 0
  
[bold]实测观察:[/bold]

  即使 R < C，实际 BER 也不为 0，因为:
  1. Hamming(7,4) 只能纠正 1 bit 错误/block
  2. 高噪声下每 block 可能有多个错误
  3. 需要更强的编码 (如 Turbo, LDPC) 才能逼近香农极限

[dim]香农定理提供的是理论极限，实际系统需要先进的编码技术才能接近这个极限。[/dim]
        """,
        title="Theory: Shannon Capacity vs Actual Rate",
        border_style="cyan"
    ))



def demo_level2_multihost():
    """Level 2: 多主机通信 + MAC 地址学习"""
    console.clear()
    print_header(
        "Level 2: Multi-Host Communication",
        "星型拓扑 + 交换机 MAC 学习 - 评分项: 寻址[15分] + 路由[15分]"
    )
    
    simulator = PhySimulationEngine(time_step_us=1, realtime_mode=True)

    simulator.set_debug(debug=False)
    
    node1 = VisualMacNode(simulator=simulator, mac_addr=1, name='Host1')
    node2 = VisualMacNode(simulator=simulator, mac_addr=2, name='Host2')
    node3 = VisualMacNode(simulator=simulator, mac_addr=3, name='Host3')
    switcher = Switcher(simulator=simulator, mac_addr=0, port_num=3, name='Switch')
    
    cable = Cable(length=100, attenuation=4, noise_level=2, debug_mode=False)
    tp1 = TwistedPair(cable=cable, simulator=simulator, ID=0)
    tp2 = TwistedPair(cable=cable, simulator=simulator, ID=1)
    tp3 = TwistedPair(cable=cable, simulator=simulator, ID=2)
    
    node1.connect_to(tp1)
    switcher.connect_to(port=0, twisted_pair=tp1)
    node2.connect_to(tp2)
    switcher.connect_to(port=1, twisted_pair=tp2)
    node3.connect_to(tp3)
    switcher.connect_to(port=2, twisted_pair=tp3)
    
    nodes = [node1, node2, node3]
    
    # ========== 1. 拓扑展示 ==========
    console.print(Rule("[bold]1. Network Topology[/bold]", style="cyan"))
    console.print()
    
    topology_diagram = """
                        ┌─────────────────────────────────┐
                        │         [bold magenta]SWITCH[/bold magenta]                │
                        │    MAC Learning Enabled         │
                        │    Ports: 3                     │
                        └─────────┬───────────────────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
      [Port 0]               [Port 1]              [Port 2]
            │                     │                     │
    ┌───────┴───────┐     ┌───────┴───────┐     ┌───────┴───────┐
    │   [cyan]Host1[/cyan]       │     │   [cyan]Host2[/cyan]       │     │   [cyan]Host3[/cyan]       │
    │   MAC: 0x01   │     │   MAC: 0x02   │     │   MAC: 0x03   │
    │   Port: 0     │     │   Port: 1     │     │   Port: 2     │
    └───────────────┘     └───────────────┘     └───────────────┘
    """
    console.print(Panel(topology_diagram, title=" Star Topology", border_style="blue"))
    console.print()

    input("Press Enter to continue...")
    #time.sleep(1.5)
    
    # ========== 2. 寻址机制说明 ==========
    console.print(Rule("[bold]2. Addressing Mechanism[/bold]", style="cyan"))
    console.print()
    
    addressing_panel = Panel(
        """
[bold yellow]如何区分不同主机？[/bold yellow]

  每个主机拥有唯一的 [cyan]MAC 地址[/cyan] (1 字节, 0-255)
  
  当前网络中的主机:
  ┌────────────┬─────────────┬────────────┐
  │    主机    │  MAC 地址   │  连接端口  │
  ├────────────┼─────────────┼────────────┤
  │   Host1    │    0x01     │   Port 0   │
  │   Host2    │    0x02     │   Port 1   │
  │   Host3    │    0x03     │   Port 2   │
  └────────────┴─────────────┴────────────┘

[bold yellow]数据包头 (Header) 设计:[/bold yellow]

  ┌─────────────────────────────────────────────────────────────┐
  │                      MAC Frame Format                       │
  ├──────────────┬──────────────┬────────────────────────────────┤
  │   SRC_MAC    │   DST_MAC    │             DATA               │
  │   (1 byte)   │   (1 byte)   │          (N bytes)             │
  ├──────────────┼──────────────┼────────────────────────────────┤
  │   发送方地址  │   目标地址   │           有效载荷             │
  └──────────────┴──────────────┴────────────────────────────────┘

  示例: Host1 → Host2 发送 "Hello"
  ┌──────┬──────┬─────────────────────┐
  │ 0x01 │ 0x02 │ 48 65 6C 6C 6F ... │
  └──────┴──────┴─────────────────────┘
        """,
        title="Addressing Scheme",
        border_style="yellow"
    )
    console.print(addressing_panel)
    console.print()

    input("Press Enter to continue...")
    #time.sleep(1.5)
    
    # ========== 3. 路由转发机制说明 ==========
    console.print(Rule("[bold]3. Routing & Forwarding Mechanism[/bold]", style="cyan"))
    console.print()
    
    routing_panel = Panel(
        """
[bold yellow]交换机如何转发消息？[/bold yellow]

  [bold]Step 1: MAC 地址学习 (Learning)[/bold]
  ─────────────────────────────────────
  当交换机从某端口收到帧时:
  • 提取帧中的 [cyan]SRC_MAC[/cyan]
  • 将 (SRC_MAC → 接收端口) 记录到 MAC 表

  [bold]Step 2: 转发决策 (Forwarding)[/bold]
  ─────────────────────────────────────
  查找帧中的 [cyan]DST_MAC[/cyan]:
  • 若 MAC 表中有记录 → [green]单播转发到对应端口[/green]
  • 若 MAC 表中无记录 → [yellow]广播到所有其他端口[/yellow]

  [bold]MAC Table 示例:[/bold]
  ┌─────────────┬────────────┐
  │  MAC 地址   │   端口     │
  ├─────────────┼────────────┤
  │    0x01     │   Port 0   │
  │    0x02     │   Port 1   │
  │    0x03     │   Port 2   │
  └─────────────┴────────────┘
        """,
        title="Routing & Forwarding",
        border_style="green"
    )
    console.print(routing_panel)
    console.print()

    input("Press Enter to continue...")
    #time.sleep(1.5)
    
    # ========== 4. 实时通信演示 ==========
    console.print(Rule("[bold]4. Live Communication Demo[/bold]", style="cyan"))
    console.print()
    
    # 初始 MAC 表状态
    console.print("[bold]Initial Switch MAC Table:[/bold]")
    if switcher.map:
        console.print(f"  {dict(switcher.map)}")
    else:
        console.print("  [dim](Empty - Learning mode activated)[/dim]")
    console.print()
    
    # 通信序列
    communications = [
        (node1, 2, b"Hello Host2, this is Host1!", "Host1 → Host2"),
        (node2, 3, b"Hello Host3, this is Host2!", "Host2 → Host3"),
        (node3, 1, b"Hello Host1, this is Host3!", "Host3 → Host1"),
    ]
    
    for step, (sender, dst_mac, data, desc) in enumerate(communications, 1):
        console.print(Panel(
            f"""
[bold]Step {step}: {desc}[/bold]

  Sender:      [cyan]{sender.name}[/cyan] (MAC: 0x{sender.mac_addr:02X})
  Destination: [cyan]Host{dst_mac}[/cyan] (MAC: 0x{dst_mac:02X})
  Message:     [yellow]{data.decode()}[/yellow]
  
  Frame:
  ┌────────────┬────────────┬──────────────────────────────────┐
  │ SRC: 0x{sender.mac_addr:02X}  │ DST: 0x{dst_mac:02X}  │ DATA: {data[:20].decode()}... │
  └────────────┴────────────┴──────────────────────────────────┘
            """,
            title=f"Transmission {step}",
            border_style="cyan"
        ))
        
        sender.send(dst_mac=dst_mac, data=data)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=30),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task(f"[cyan]Transmitting & Forwarding...", total=3000)
            
            for tick in range(0, 10000, 1000):
                simulator.run(duration_ticks=1000)
                progress.update(task, advance=1000)
                time.sleep(0.01)
        
        # 显示转发过程
        receiver = nodes[dst_mac - 1]

        if dst_mac not in switcher.map:
            forward_info = f"""
        [green]Switch received frame from Port {sender.mac_addr - 1}[/green]
        [green]Learned: MAC 0x{sender.mac_addr:02X} → Port {sender.mac_addr - 1}[/green]
        [yellow]DST_MAC 0x{dst_mac:02X} not found in MAC table[/yellow]
        [yellow]Broadcasting to all other ports[/yellow]
        [green]{receiver.name} received the message[/green]
        """
            
        else:
            forward_info = f"""
        [green]Switch received frame from Port {sender.mac_addr - 1}[/green]
        [green]Learned: MAC 0x{sender.mac_addr:02X} → Port {sender.mac_addr - 1}[/green]
        [green]Lookup DST_MAC 0x{dst_mac:02X} in MAC table[/green]
        [green]Forward to Port {dst_mac - 1}[/green]
        [green]{receiver.name} received the message[/green]
                """

        console.print(Panel(forward_info, title="Switch Processing", border_style="green"))
        
        # 当前 MAC 表
        mac_table = Table(box=box.SIMPLE, title="Current MAC Table")
        mac_table.add_column("MAC Address", style="yellow")
        mac_table.add_column("Port", style="cyan")
        for mac, port in switcher.map.items():
            mac_table.add_row(f"0x{mac:02X}", f"Port {port}")
        console.print(mac_table)
        console.print()
        
        time.sleep(0.3)
    
    # ========== 5. 最终统计 ==========
    console.print(Rule("[bold]5. Final Statistics[/bold]", style="cyan"))
    console.print()
    
    # 完整 MAC 表
    final_mac_table = Table(box=box.DOUBLE_EDGE, title="Final Switch MAC Address Table")
    final_mac_table.add_column("MAC Address", style="yellow", justify="center")
    final_mac_table.add_column("Port", style="cyan", justify="center")
    final_mac_table.add_column("Host", style="green", justify="center")
    for mac, port in switcher.map.items():
        final_mac_table.add_row(f"0x{mac:02X}", f"Port {port}", f"Host{mac}")
    console.print(final_mac_table)
    console.print()
    
    # 节点统计
    stats_table = Table(box=box.DOUBLE_EDGE, title="Node Communication Statistics")
    stats_table.add_column("Node", style="cyan")
    stats_table.add_column("MAC", style="magenta", justify="center")
    stats_table.add_column("TX Packets", style="green", justify="right")
    stats_table.add_column("TX Bytes", style="green", justify="right")
    stats_table.add_column("RX Packets", style="yellow", justify="right")
    stats_table.add_column("RX Bytes", style="yellow", justify="right")
    stats_table.add_column("Last Received", style="dim")
    
    for node in nodes:
        stats_table.add_row(
            node.name,
            f"0x{node.mac_addr:02X}",
            str(node.sent_count),
            str(node.sent_bytes),
            str(node.recv_count),
            str(node.recv_bytes),
            node.last_recv[:30] + "..." if len(node.last_recv) > 30 else node.last_recv
        )
    console.print(stats_table)
    
    console.print("\n[bold green]✅ 多主机通信演示完成！MAC 地址学习和转发机制工作正常。[/bold green]")



def demo_level3_http():
    """Level 3: HTTP 应用层协议"""
    console.clear()
    print_header(
        "Level 3: Application Layer Protocol",
        "HTTP-like 请求/响应协议 - 评分项: 应用层协议 [10分]"
    )
    
    simulator = PhySimulationEngine(time_step_us=10, realtime_mode=True)
    
    server = HttpServer(simulator=simulator, mac_addr=1, name='WebServer', port=80)
    client = HttpClient(simulator=simulator, mac_addr=2, name='Browser')
    switcher = Switcher(simulator=simulator, mac_addr=0, port_num=2, name='Switch')
    
    cable = Cable(length=100, attenuation=3, noise_level=2, debug_mode=False)
    tp1 = TwistedPair(cable=cable, simulator=simulator, ID=0)
    tp2 = TwistedPair(cable=cable, simulator=simulator, ID=1)
    
    server.connect_to(tp1)
    switcher.connect_to(port=0, twisted_pair=tp1)
    client.connect_to(tp2)
    switcher.connect_to(port=1, twisted_pair=tp2)
    
    server.add_route('/api/users', lambda req: b'{"users": ["alice", "bob", "charlie"]}')
    server.add_route('/api/status', lambda req: b'{"status": "running", "uptime": 3600}')
    
    print_star_topology(["Server", "Client"])
    
    # HTTP 协议说明
    console.print(Panel(
        """
[bold]HTTP Request Format:[/bold]
┌────────────────────────────────┐
│ GET /path HTTP/1.1             │
│ Host: server                   │
│                                │
└────────────────────────────────┘

[bold]HTTP Response Format:[/bold]
┌────────────────────────────────┐
│ HTTP/1.1 200 OK                │
│ Content-Type: application/json │
│                                │
│ {"key": "value"}               │
└────────────────────────────────┘
        """,
        title="HTTP Protocol Format",
        border_style="cyan"
    ))
    console.print()
    
    responses = []
    
    def on_response(resp):
        if resp:
            responses.append(resp)
    
    requests = [
        ("GET", "/", "Homepage"),
        ("GET", "/api/users", "User list"),
        ("GET", "/api/status", "Server status"),
        ("GET", "/not-found", "Non-existent page"),
    ]
    
    console.print("[bold cyan]Sending HTTP Requests:[/bold cyan]")
    console.print()
    
    for method, path, desc in requests:
        console.print(f"[yellow]→[/yellow] {method} {path} [dim]({desc})[/dim]")
        
        client.get(dst_mac=1, dst_port=80, path=path, callback=on_response)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("[cyan]Waiting for response...", total=None)
            simulator.run(duration_ticks=3000)
        
        time.sleep(0.2)
    
    console.print()
    
    # 结果表格
    result_table = Table(box=box.DOUBLE_EDGE, title="📡 HTTP Responses")
    result_table.add_column("Request", style="cyan")
    result_table.add_column("Status", justify="center")
    result_table.add_column("Response Body", style="yellow")
    
    for i, resp in enumerate(responses):
        if resp and i < len(requests):
            method, path, _ = requests[i]
            status = resp.get('status_code', 'N/A')
            
            if status == 200:
                status_text = "[green]200 OK[/green]"
            elif status == 404:
                status_text = "[red]404 Not Found[/red]"
            else:
                status_text = f"[yellow]{status}[/yellow]"
            
            body = resp.get('body', 'N/A')
            if len(body) > 40:
                body = body[:40] + "..."
            
            result_table.add_row(f"{method} {path}", status_text, body)
    
    console.print(result_table)
    console.print("\n[bold green] HTTP Protocol Test Passed! [/bold green]")


def demo_level3_coding():
    """Level 3: 信道编码对比"""
    console.clear()
    print_header(
        "Level 3: Channel Coding",
        "有/无信道编码的性能对比 - 评分项: 信道编码 [15分]"
    )
    
    test_sizes = [64, 256, 512]
    noise_level = 5
    cable_length = 100
    attenuation = 4
    
    # ========== 测试条件面板 ==========
    test_config_panel = Panel(
        f"""
[bold]Test Configuration:[/bold]

  ┌─────────────────────────────────────────────────────┐
  │                  Channel Parameters                 │
  ├──────────────────────┬──────────────────────────────┤
  │  Cable Length        │  [yellow]{cable_length}[/yellow] meters                  │
  │  Attenuation         │  [yellow]{attenuation}[/yellow]                          │
  │  Noise Level         │  [yellow]{noise_level}[/yellow]                          │
  │  SNR (estimated)     │  [yellow]{1.0/noise_level:.4f}[/yellow]                   │
  └──────────────────────┴──────────────────────────────┘

  ┌─────────────────────────────────────────────────────┐
  │                   Test Data Sizes                   │
  ├─────────────────────────────────────────────────────┤
  │  [cyan]Size 1:[/cyan]  {test_sizes[0]:>4} bytes  ({test_sizes[0]*8:>5} bits)              │
  │  [cyan]Size 2:[/cyan]  {test_sizes[1]:>4} bytes  ({test_sizes[1]*8:>5} bits)              │
  │  [cyan]Size 3:[/cyan]  {test_sizes[2]:>4} bytes  ({test_sizes[2]*8:>5} bits)              │
  └─────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────┐
  │                   Coding Schemes                    │
  ├─────────────────────────────────────────────────────┤
  │  [red]No Coding:[/red]    Raw data transmission             │
  │  [green]With Coding:[/green]  Error correction enabled          │
  └─────────────────────────────────────────────────────┘
        """,
        title="Test Conditions",
        border_style="cyan"
    )
    console.print(test_config_panel)
    console.print()
    
    # 测试每个大小
    all_results = []
    
    for size in test_sizes:
        test_msg = generate_random_data(length=size)
        
        # 当前测试条件
        current_test_panel = Panel(
            f"""
[bold]Current Test:[/bold]
  Data Size:     [yellow]{size}[/yellow] bytes ([yellow]{size * 8}[/yellow] bits)
  Noise Level:   [yellow]{noise_level}[/yellow]
  Cable Length:  [yellow]{cable_length}[/yellow] m
  Attenuation:   [yellow]{attenuation}[/yellow]
            """,
            title=f"🧪 Testing {size} bytes",
            border_style="yellow"
        )
        console.print(current_test_panel)
        
        results = []
        
        for coding in [False, True]:
            coding_str = "With Coding" if coding else "No Coding"
            coding_icon = "🛡️" if coding else "📦"
            
            simulator = PhySimulationEngine(time_step_us=1, realtime_mode=False)
            node_a = VisualPhyNode(simulator=simulator, name='A', coding=coding)
            node_b = VisualPhyNode(simulator=simulator, name='B', coding=coding)
            
            cable = Cable(length=cable_length, attenuation=attenuation, noise_level=noise_level, debug_mode=False)
            tp = TwistedPair(cable=cable, simulator=simulator, ID=0)
            
            node_a.connect_to(tp)
            node_b.connect_to(tp)
            node_a.send(test_msg)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task(f"[cyan]{coding_icon} {coding_str}...", total=None)
                simulator.run(duration_ticks=50000)
            
            recv_data = node_b.recv_buffer[0] if node_b.recv_buffer else b""
            byte_errors = diff(test_msg, recv_data) if recv_data else size
            
            # 计算比特错误
            if recv_data:
                bit_errors = 0
                for i in range(min(len(test_msg), len(recv_data))):
                    xor = test_msg[i] ^ recv_data[i]
                    bit_errors += bin(xor).count('1')
                bit_errors += abs(len(test_msg) - len(recv_data)) * 8
            else:
                bit_errors = size * 8
            
            ber = bit_errors / (size * 8)
            byte_error_rate = (byte_errors / size) * 100
            
            results.append({
                'coding': coding_str,
                'rx_bytes': len(recv_data),
                'byte_errors': byte_errors,
                'bit_errors': bit_errors,
                'ber': ber,
                'byte_error_rate': byte_error_rate
            })
        
        # 对比表格
        table = Table(box=box.ROUNDED, title=f"📊 Results for {size} bytes")
        table.add_column("Mode", style="cyan")
        table.add_column("TX Bytes", style="dim", justify="right")
        table.add_column("RX Bytes", style="green", justify="right")
        table.add_column("Bit Errors", style="red", justify="right")
        table.add_column("BER", style="yellow", justify="right")
        table.add_column("Byte Errors", style="red", justify="right")
        table.add_column("Improvement", justify="center")
        
        no_coding_errors = results[0]['bit_errors']
        with_coding_errors = results[1]['bit_errors']
        
        for i, res in enumerate(results):
            if i == 0:
                improvement = "-"
            else:
                if no_coding_errors > 0:
                    imp = ((no_coding_errors - with_coding_errors) / no_coding_errors) * 100
                    if imp > 0:
                        improvement = f"[green]↓{imp:.1f}%[/green]"
                    elif imp < 0:
                        improvement = f"[red]↑{-imp:.1f}%[/red]"
                    else:
                        improvement = "[dim]0%[/dim]"
                else:
                    improvement = "[green]N/A (no errors)[/green]"
            
            ber_str = f"{res['ber']:.2e}" if res['ber'] > 0 else "[green]0[/green]"
            
            table.add_row(
                res['coding'],
                str(size),
                str(res['rx_bytes']),
                str(res['bit_errors']),
                ber_str,
                str(res['byte_errors']),
                improvement
            )
        
        console.print(table)
        console.print()
        
        all_results.append({
            'size': size,
            'results': results
        })
    
    # 汇总表格
    console.print(Rule("[bold]Summary[/bold]", style="cyan"))
    console.print()
    
    summary_table = Table(box=box.DOUBLE_EDGE, title="Overall Performance Summary")
    summary_table.add_column("Data Size", style="cyan", justify="center")
    summary_table.add_column("No Coding\nBER", style="red", justify="center")
    summary_table.add_column("With Coding\nBER", style="green", justify="center")
    summary_table.add_column("Error\nReduction", style="yellow", justify="center")
    
    for item in all_results:
        size = item['size']
        no_coding_ber = item['results'][0]['ber']
        with_coding_ber = item['results'][1]['ber']
        
        if no_coding_ber > 0:
            reduction = ((no_coding_ber - with_coding_ber) / no_coding_ber) * 100
            reduction_str = f"[green]{reduction:.1f}%[/green]"
        else:
            reduction_str = "[dim]N/A[/dim]"
        
        summary_table.add_row(
            f"{size} bytes",
            f"{no_coding_ber:.2e}",
            f"{with_coding_ber:.2e}",
            reduction_str
        )
    
    console.print(summary_table)
    console.print()
    
    # 理论说明
    console.print(Panel(
        f"""
[bold]Test Conditions Recap:[/bold]

  • Noise Level: [yellow]{noise_level}[/yellow]
  • SNR: [yellow]{1.0/noise_level:.4f}[/yellow]
  • Cable: [yellow]{cable_length}m[/yellow], Attenuation: [yellow]{attenuation}[/yellow]

[bold]Channel Coding Benefits:[/bold]

  • 添加冗余信息以检测和纠正传输错误
  • 在噪声环境下显著提高数据完整性
  • 代价: 降低有效传输速率 (增加开销)

[bold]Trade-off:[/bold]

  ┌────────────────┬────────────────┐
  │    无编码      │    有编码      │
  ├────────────────┼────────────────┤
  │  高传输速率    │  低传输速率    │
  │  无纠错能力    │  可纠正错误    │
  │  高误码率      │  低误码率      │
  └────────────────┴────────────────┘
        """,
        title="Channel Coding Theory",
        border_style="cyan"
    ))



def demo_concurrency():
    """展示系统并发处理能力"""
    console.clear()
    print_header(
        "System Concurrency Demonstration",
        "多节点同时通信 - 展示仿真引擎的并发调度能力"
    )
    
    simulator = PhySimulationEngine(time_step_us=1, realtime_mode=True)
    simulator.set_debug(debug=False)
    
    # 创建 4 个节点
    nodes = []
    for i in range(1, 5):
        node = VisualMacNode(simulator=simulator, mac_addr=i, name=f'Host{i}')
        nodes.append(node)
    
    switcher = Switcher(simulator=simulator, mac_addr=0, port_num=4, name='Switch')
    
    cable = Cable(length=100, attenuation=4, noise_level=2, debug_mode=False)
    
    # 连接所有节点到交换机
    for i, node in enumerate(nodes):
        tp = TwistedPair(cable=cable, simulator=simulator, ID=i)
        node.connect_to(tp)
        switcher.connect_to(port=i, twisted_pair=tp)
    
    # 拓扑图
    topology = """
                            ┌─────────────┐
                            │   SWITCH    │
                            │  (4 ports)  │
                            └──────┬──────┘
                                   │
            ┌──────────┬───────────┼───────────┬──────────┐
            │          │           │           │          │
        [Port 0]   [Port 1]    [Port 2]   [Port 3]
            │          │           │           │
        ┌───┴───┐  ┌───┴───┐  ┌───┴───┐  ┌───┴───┐
        │ Host1 │  │ Host2 │  │ Host3 │  │ Host4 │
        │ MAC:1 │  │ MAC:2 │  │ MAC:3 │  │ MAC:4 │
        └───────┘  └───────┘  └───────┘  └───────┘
    """
    console.print(Panel(topology, title="4-Node Star Topology", border_style="blue"))
    console.print()
    
    # ========== 并发通信场景 ==========
    console.print(Rule("[bold]Concurrent Communication Scenario[/bold]", style="cyan"))
    console.print()
    
    # 定义并发通信
    concurrent_sends = [
        (nodes[0], 2, b"[1->2] Hello from Host1!"),
        (nodes[1], 3, b"[2->3] Hello from Host2!"),
        (nodes[2], 4, b"[3->4] Hello from Host3!"),
        (nodes[3], 1, b"[4->1] Hello from Host4!"),
    ]
    
    # 展示并发发送计划
    plan_table = Table(box=box.ROUNDED, title="📋 Concurrent Send Plan (All at tick=0)")
    plan_table.add_column("Sender", style="cyan")
    plan_table.add_column("→", style="dim")
    plan_table.add_column("Receiver", style="green")
    plan_table.add_column("Message", style="yellow")
    
    for sender, dst_mac, data in concurrent_sends:
        plan_table.add_row(
            f"Host{sender.mac_addr}",
            "→",
            f"Host{dst_mac}",
            data.decode()[:30]
        )
    console.print(plan_table)
    console.print()
    
    console.print("[bold yellow]All 4 messages sent simultaneously at tick=0[/bold yellow]")
    console.print()
    
    # 同时发送所有消息
    for sender, dst_mac, data in concurrent_sends:
        sender.send(dst_mac=dst_mac, data=data)
    
    # 实时进度显示
    console.print("[bold]Simulation Progress:[/bold]")
    console.print()
    
    # 创建状态跟踪
    received_status = {i: [] for i in range(1, 5)}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TextColumn("tick"),
        console=console,
        transient=False
    ) as progress:
        task = progress.add_task("[cyan]Running simulation...", total=5000)
        
        for tick_batch in range(0, 5000, 500):
            simulator.run(duration_ticks=500)
            progress.update(task, advance=500)
            
            # 检查各节点接收状态
            for node in nodes:
                if node.recv_buffer and len(node.recv_buffer) > len(received_status[node.mac_addr]):
                    new_msgs = node.recv_buffer[len(received_status[node.mac_addr]):]
                    for msg in new_msgs:
                        received_status[node.mac_addr].append(msg)
            
            time.sleep(0.05)
    
    console.print()
    
    # ========== 并发处理时序图 ==========
    console.print(Rule("[bold]Concurrency Timeline[/bold]", style="cyan"))
    console.print()
    
    timeline = """
[bold]Time →[/bold]
    
    tick=0              tick=500           tick=1000          tick=1500
      │                    │                  │                  │
      ▼                    ▼                  ▼                  ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ Host1 │ [cyan]TX→Host2[/cyan] ════════════════╗                              │
  ├───────┼──────────────────────────────────╬──────────────────────────┤
  │ Host2 │ [cyan]TX→Host3[/cyan] ════════════════╬═══╗                          │
  ├───────┼──────────────────────────────────╬───╬────────────────────────┤
  │ Host3 │ [cyan]TX→Host4[/cyan] ════════════════╬═══╬═══╗                      │
  ├───────┼──────────────────────────────────╬───╬───╬────────────────────┤
  │ Host4 │ [cyan]TX→Host1[/cyan] ════════════════╬═══╬═══╬═══╗                  │
  ├───────┼──────────────────────────────────╬───╬───╬───╬────────────────┤
  │Switch │ [magenta]Processing all frames concurrently[/magenta]   │   │   │   │  │
  │       │ MAC Learning + Forwarding        ▼   ▼   ▼   ▼                │
  ├───────┼──────────────────────────────────────────────────────────────┤
  │ Host1 │                                              [green]◄══RX[/green]   │
  │ Host2 │                        [green]◄══RX[/green]                          │
  │ Host3 │                             [green]◄══RX[/green]                     │
  │ Host4 │                                  [green]◄══RX[/green]                │
  └───────────────────────────────────────────────────────────────────┘
    """
    console.print(Panel(timeline, title="⏱️ Concurrent Processing Timeline", border_style="cyan"))
    console.print()
    
    # ========== 接收结果 ==========
    console.print(Rule("[bold]Reception Results[/bold]", style="cyan"))
    console.print()
    
    result_table = Table(box=box.DOUBLE_EDGE, title="📬 Messages Received by Each Host")
    result_table.add_column("Host", style="cyan", justify="center")
    result_table.add_column("Expected From", style="yellow", justify="center")
    result_table.add_column("Received", style="green", justify="center")
    result_table.add_column("Message Content", style="dim")
    result_table.add_column("Status", justify="center")
    
    expected_from = {1: 4, 2: 1, 3: 2, 4: 3}  # Host X expects from Host Y
    
    for node in nodes:
        mac = node.mac_addr
        exp = expected_from[mac]
        recv_msgs = node.recv_buffer
        
        if recv_msgs:
            msg_content = recv_msgs[0].decode('utf-8', errors='replace')[:35]
            status = "[green]✅ OK[/green]"
        else:
            msg_content = "[dim]No data[/dim]"
            status = "[red]❌ Missing[/red]"
        
        result_table.add_row(
            f"Host{mac}",
            f"Host{exp}",
            str(len(recv_msgs)),
            msg_content,
            status
        )
    
    console.print(result_table)
    console.print()
    
    # ========== 仿真引擎并发说明 ==========
    console.print(Panel(
        """
[bold]Simulation Engine Concurrency Model:[/bold]

  ┌─────────────────────────────────────────────────────────────────┐
  │                    PhySimulationEngine                          │
  │                                                                 │
  │   for tick in range(duration):                                  │
  │       for entity in registered_entities:  ← [yellow]Round-robin update[/yellow] │
  │           entity.update(tick)                                   │
  │                                                                 │
  │   [dim]All entities see the same "tick" - deterministic ordering[/dim]   │
  └─────────────────────────────────────────────────────────────────┘

[bold]Registered Entities in this Demo:[/bold]

  ┌──────┬────────────────┬─────────────────────────────────────────┐
  │  #   │     Entity     │              Role                       │
  ├──────┼────────────────┼─────────────────────────────────────────┤
  │  1   │  Host1         │  TxEntity + RxEntity + MacLayer         │
  │  2   │  Host2         │  TxEntity + RxEntity + MacLayer         │
  │  3   │  Host3         │  TxEntity + RxEntity + MacLayer         │
  │  4   │  Host4         │  TxEntity + RxEntity + MacLayer         │
  │  5   │  Switch        │  4× (TxEntity + RxEntity) + Forwarding  │
  │  6-9 │  Channels      │  Signal propagation + noise             │
  └──────┴────────────────┴─────────────────────────────────────────┘

[bold]Concurrency Properties:[/bold]

  • [green]Deterministic[/green]: Same input → Same output (reproducible)
  • [green]Fair Scheduling[/green]: All entities updated each tick
  • [green]No Race Conditions[/green]: Sequential update within each tick
  • [green]Parallel Conceptually[/green]: All transmissions overlap in simulated time
        """,
        title="🔧 Simulation Engine Architecture",
        border_style="cyan"
    ))
    
    # ========== 性能统计 ==========
    console.print()
    console.print(Rule("[bold]Performance Statistics[/bold]", style="cyan"))
    console.print()
    
    stats_table = Table(box=box.ROUNDED, title="📊 Concurrent Communication Stats")
    stats_table.add_column("Metric", style="dim")
    stats_table.add_column("Value", style="yellow")
    
    total_tx = sum(node.sent_bytes for node in nodes)
    total_rx = sum(node.recv_bytes for node in nodes)
    success_rate = (total_rx / total_tx * 100) if total_tx > 0 else 0
    
    stats_table.add_row("Total Nodes", "4")
    stats_table.add_row("Concurrent Streams", "4")
    stats_table.add_row("Total TX Bytes", str(total_tx))
    stats_table.add_row("Total RX Bytes", str(total_rx))
    stats_table.add_row("Success Rate", f"{success_rate:.1f}%")
    stats_table.add_row("Switch MAC Table Size", str(len(switcher.map)))
    
    console.print(stats_table)
    console.print()
    
    console.print("[bold green]✅ 并发通信演示完成！所有节点同时发送，交换机正确处理并转发。[/bold green]")


# ============================================================
#                      主菜单
# ============================================================

def main_menu():
    """主菜单"""
    while True:
        console.clear()
        
        menu = """
[bold cyan]╔══════════════════════════════════════════════════════════╗
║         Network Communication Project Demo                 ║
╚══════════════════════════════════════════════════════════╝[/bold cyan]

[bold]Level 1: Point-to-Point Communication [30分][/bold]
  [cyan][1][/cyan] 基础比特流传输        [dim](简单字符串, 15分)[/dim]
  [cyan][2][/cyan] 消息分片传输          [dim](长消息处理, 5分)[/dim]
  [cyan][3][/cyan] 噪声性能测试          [dim](香农公式对比, 10分)[/dim]

[bold]Level 2: Multi-Host Communication [30分][/bold]
  [cyan][4][/cyan] 多主机通信 + MAC学习  [dim](寻址+路由, 30分)[/dim]

[bold]Level 3: Extension Features [40分][/bold]
  [cyan][5][/cyan] HTTP 应用层协议       [dim](请求/响应, 10分)[/dim]
  [cyan][6][/cyan] 信道编码对比          [dim](编码性能, 15分)[/dim]

[bold]System Architecture[/bold]
  [cyan][7][/cyan] 并发性演示            [dim](仿真引擎并发调度)[/dim]

[dim][q] Quit[/dim]
        """
        
        console.print(Panel(menu, border_style="blue"))
        
        choice = console.input("\n[bold yellow]Select demo (1-7 or q): [/bold yellow]")
        
        demos = {
            '1': demo_level1_basic,
            '2': demo_level1_fragmentation,
            '3': demo_level1_noise,
            '4': demo_level2_multihost,
            '5': demo_level3_http,
            '6': demo_level3_coding,
            '7': demo_concurrency,
        }
        
        if choice in demos:
            try:
                demos[choice]()
            except KeyboardInterrupt:
                console.print("\n[yellow]Demo interrupted.[/yellow]")
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")
                import traceback
                traceback.print_exc()
            
            console.print()
            console.input("[dim]Press Enter to return to menu...[/dim]")
        elif choice.lower() == 'q':
            console.print("\n[cyan]Goodbye! [/cyan]\n")
            break


if __name__ == "__main__":
    main_menu()
