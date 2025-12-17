# Data Communication and Networks - Course Project

Final Project截止日期：12.18 14:00。注意无论你的答辩时间是15周还是16周，截止日期都是12.18. 

你需要上传 代码、展示用的PPT和其他材料至BB。

Final Project需要撰写报告，报告截止时间：12.28 23:59。

## Presentation Guide

你的展示应该涵盖以下关键部分，以确保能够展示出你们在各个 Level 中的工作成果。

展示时间要求：5分钟。请合理规划你的展示内容，着重强调**你做了什么**以及展示**你做了什么**。

我们推荐你的Presentation至少包括以下部分：

### Level 1: 点对点通信 (Point-to-Point Communication) - [30分]

这部分重点展示基础通信链路的实现。TIPS: 你可以使用随机生成的1/0 binary序列来模拟上层比特流，也可以传输任意的字符串编码后额比特流。简单介绍你们的实现方法。

- 录屏展示Level 1的完整的比特流传输过程。
- 录屏展示Level 1的消息分片传输功能。
- 展示在有噪声 (`noise_level > 0`) 情况下系统的表现。
- 观察不同噪声下的系统传输速率，跟香农公式进行比较。(B log2(1+SNR))

### Level 2: 多主机通信 (Multi-Host Communication) - [30分]

这部分重点展示网络拓扑和交换/路由逻辑。简单介绍你们的实现方法。

- 录屏展示针对简单拓扑的多主机通信的完整过程。
- **寻址机制 (Addressing)**:
  - 你们如何区分不同的主机？
  - 数据包头 (Header) 是如何设计的？
- **路由与转发 (Routing & Forwarding)**:
  - 中心节点 (Switch/Router) 如何根据地址将消息转发给正确的目标主机？

### Level 3: 扩展功能 (Extension Features) - [最高 40分]

选择你们实现的扩展功能进行详细介绍。需要基于Level 2完整实现。需要**现场**演示功能以及简单解释你们的实现方法。

* **传输层 (Transport Layer)**: 可靠传输 (ACK/NACK)、序列号、超时重传、流控等。
* **信道编码 (Channel Coding)**: 纠错码、误码率 vs 纠错率的性能测试；有无信道编码的信息传输速率测试。
* **应用层协议 (Application Layer)**: 自定义协议设计 (HTTP/FTP-like)、请求-响应模式。需要基于你自己的 level 1/level 2 完整实现一个有实际意义的应用层协议。
* **性能优化 (Performance)**: 多种调制方式对比 (ASK/FSK/PSK)、抗噪性能分析。
* **无线通信 (Wireless)**: 无线信道模拟、MIMO、波束成形等。
* **并发处理 (Concurrency)**: 多线程处理、高并发访问支持。

### 遇到的挑战与解决方案 (Challenges & Solutions)

* 在开发过程中遇到了什么困难？
* 你们是如何解决这些问题的

## Project Requirements

### Level 1: Point-to-Point Communication (30 points)

**Scenario**: Two hosts connected directly via cable

**Requirements**:

1. Implement digital signal generation (convert data to bit stream)
2. Implement modulator (convert digital signal to analog signal)
3. Use the provided `Cable` class to transmit signals
4. Implement demodulator (recover digital signal from analog signal)
5. Implement data recovery (convert bit stream back to original data)

**Flow**:

```
Host A: Data → Bit Stream → Modulation → Analog Signal → Cable → Analog Signal → Demodulation → Bit Stream → Data :Host B
```

**Grading Criteria**:

- Successfully transmit simple strings (15 points)
- Handle longer messages (5 points). This means your protocol supports **slicing**, i.e., to slice a packet into mutiple packets, and compose them into one packet.
- Basic error detection mechanism (10 points)

---

### Level 2: Multi-Host Communication (30 points)

**Scenario**: N hosts in the same network. You need to use switches or routers for multi-user communication, instead of N(N-1) direct connections (full mesh topology).

**Topology**: Use a Star Topology where all hosts connect to a central Switch/Router. Do not use a Full Mesh topology where every host has a direct cable to every other host.

**Forwarding**: The central device (Switch/Router) must receive signals from a source host and forward them to the intended destination host based on the addressing scheme you design.

**Requirements**:

1. Based on Level 1, design a mechanism to distinguish different hosts
2. Implement addressing mechanism (how to specify target host)
3. Implement routing/forwarding mechanism (how messages reach their destination)
4. Handle simultaneous transmissions from multiple hosts. This means the central device can handle multiple access. As a counterpart, a central device with inability of mutiple access will disgard all messages received from other hosts (or ignore it), when it is processing message from specific host. The concurrency processing can be assumed as a part of **Extension Feature.**

**Grading Criteria**:

- Distinguish different hosts (15 points)
- Correctly route to target host (15 points)

---

### Level 3: Extension Features (at most 40 points)

Choose from the following features to implement (can select multiple):

#### Transport Layer (15 points)

- Implement reliable transport (ACK/NACK)
- Implement sequence numbers
- Implement timeout retransmission
- Implement flow control

#### Channel Coding (15 points)

- Implement Hamming code or other error correction codes
- Implement CRC checksum
- Correct transmission errors
- Performance testing (error rate vs correction rate)

#### Application Layer Protocol (10 points)

- Design application layer protocol (e.g., HTTP, FTP, etc.)
- Implement request-response pattern
- Implement protocol parsing

#### Performance Optimization (10 points)

- Implement multiple modulation schemes (ASK, FSK, PSK, etc.)
- Compare performance of different modulation schemes

#### Wireless Communication (10 points)

- Use wireless channel (instead of cable).
- MIMO.
- Precoding & combining (beamforming).

#### Concunrrency (10 points)

- Concurrency means your network can handle massive access and data transmission. Messages won't blcok each other.
- You should use multiple threads to do this

## Provided Tools

### Cable Class (cable.py)

This is the only infrastructure provided by the course, simulating the physical transmission medium. See, **cables are everywhere. So implement it (call it) everywhere.**

**Main Methods**:

```python
cable = Cable(length=100, attenuation=0.1, noise_level=0.05, debug_mode=True)

# Transmit signal
received_signal = cable.transmit(signal)

# Get propagation delay
delay = cable.get_propagation_delay()

# Plot waveforms (debug mode)
cable.plot_signals()

# Get statistics
stats = cable.get_signal_stats()
```

**Parameter Descriptions**:

- `length`: Cable length (meters)
- `attenuation`: Attenuation coefficient
- `noise_level`: Noise level (higher means more noise)
- `debug_mode`: Whether to enable debug mode (display waveforms)

---

## Getting Started

### 1. Understand the Cable Class

Run the provided example:

```bash
python cable.py
```

### 2. Implement Level 1

Start with the simplest approach:

**Step 1**: Implement a simple modulator

```python
def modulate(bits):
    # Convert bits to analog signal
    # Hint: Use high level for 1, low level for 0
    pass
```

**Step 2**: Implement demodulator

```python
def demodulate(signal):
    # Recover bits from analog signal
    pass
```

**Step 3**: Test

```python
# Sender
message = "Hello"
bits = string_to_bits(message)
signal = modulate(bits)

# Transmission
cable = Cable()
received_signal = cable.transmit(signal)

# Receiver
received_bits = demodulate(received_signal)
received_message = bits_to_string(received_bits)

print(f"Sent: {message}")
print(f"Received: {received_message}")
```

## Grading Criteria

- **Level 1 (30 points)**: Basic communication functionality
- **Level 2 (30 points)**: Multi-host addressing and routing
- **Level 3 (40 points)**: Extension features (multiple options available)

**Total Score**: 100 points (60 points base + 40 points extensions)

---

## FAQ

**Q1: How to generate analog signals?**
A: The simplest way is using numpy arrays. Use high level (e.g., 1.0) for 1 and low level (e.g., 0.0) for 0. Each bit can be represented by multiple sample points.

For example, to generate a sine wave (analog signal):

```python
import numpy as np

# Parameters
duration = 1.0  # seconds
sample_rate = 1000  # Hz
frequency = 5  # Hz (5 cycles per second)

# Time array
t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)

# Generate sine wave
analog_signal = np.sin(2 * np.pi * frequency * t)
```

**Q2: Will noise affect communication?**
A: Yes! That's why you need to design good modulation schemes and error detection mechanisms. Start with low noise for testing, then gradually increase it.

**Q3: Must I implement all levels?**
A: No. Levels 1 and 2 are foundational (60 points), Level 3 is for extensions (40 points). Choose based on your time and capability.

**Q4: Can I use third-party libraries?**
A: You can use common libraries like numpy and matplotlib, but core modulation/demodulation and protocol design must be implemented by yourself.

**Q5: How to debug?**
A: Use Cable's debug_mode to view signal waveforms. You can also use print statements for intermediate results.

Happy coding! If you have questions, please contact the TA (yaolj2024@mail.sustech.edu.cn). We prioritize these who draw their issues in GitHub.