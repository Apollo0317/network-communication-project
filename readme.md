# Network Communication Simulation Project

A comprehensive network protocol stack simulation system built from scratch, implementing physical layer to application layer with realistic signal propagation and protocol behaviors.

## 🌟 Features

- **Full Protocol Stack Implementation**: PHY → MAC → TCP → HTTP
- **Tick-based Simulation Engine**: Precise timing control at microsecond granularity
- **Realistic Physical Layer**: Manchester encoding, signal attenuation, noise simulation
- **MAC Layer**: CSMA/CD collision detection, Ethernet-like framing, switch forwarding
- **Transport Layer**: Go-Back-N (GBN) and Selective Repeat (SR) ARQ protocols
- **Application Layer**: HTTP client/server with non-blocking socket API
- **Modular Architecture**: Easy to extend with new protocols or components

## 📁 Project Structure

```
network-communication-project/
├── core/                       # Core simulation framework
│   ├── simulator.py            # Game-loop simulation engine
│   └── ProtocolStack.py        # Protocol layer base class
│
├── phy/                        # Physical Layer
│   ├── PhyLayer.py             # PHY layer implementation
│   ├── TwistedPair.py          # Twisted pair cable simulation
│   └── Cable.py                # Cable characteristics (attenuation, noise)
│
├── mac/                        # MAC Layer
│   ├── MacLayer.py             # MAC protocol with CSMA/CD
│   └── Switcher.py             # L2 switch with MAC learning
│
├── tcp/                        # Transport Layer
│   ├── TransportLayer.py       # GBN & SR ARQ implementations
│   └── socket.py               # POSIX-like socket API
│
├── app/                        # Application Layer
│   ├── client.py               # HTTP client node
│   ├── server.py               # HTTP server node
│   └── test_http.py            # HTTP communication tests
│
└── test/                       # Test suites
    └── test_stack.py           # Protocol stack integration tests
```

## 🏗️ Architecture

### Simulation Engine (Game Loop Pattern)

```
┌─────────────────────────────────────────────────────────┐
│                  PhySimulationEngine                     │
│  ┌─────────────────────────────────────────────────┐    │
│  │  for tick in range(duration):                   │    │
│  │      for entity in entities:                    │    │
│  │          entity.update(tick)                    │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
   ┌─────────┐       ┌─────────┐       ┌───────────┐
   │  Node1  │       │  Node2  │       │  Switcher │
   │ (Client)│       │ (Server)│       │           │
   └─────────┘       └─────────┘       └───────────┘
```

### Protocol Stack (Layered Architecture)

```
┌────────────────────────────────────────┐
│           Application Layer            │
│         (HttpClient/HttpServer)        │
├────────────────────────────────────────┤
│           Transport Layer              │
│      (GBN/SR ARQ + Socket API)         │
├────────────────────────────────────────┤
│              MAC Layer                 │
│     (CSMA/CD, Framing, Switching)      │
├────────────────────────────────────────┤
│            Physical Layer              │
│   (Manchester Coding, Signal Prop.)    │
├────────────────────────────────────────┤
│          Transmission Medium           │
│     (TwistedPair + Cable Model)        │
└────────────────────────────────────────┘
```

## 🔧 Key Components

### 1. Physical Layer (`phy/`)

- **Manchester Encoding**: Self-clocking line code for reliable bit synchronization
- **Signal Propagation**: Realistic delay based on cable length and signal speed
- **Cable Model**: Configurable attenuation and Gaussian noise injection
- **Twisted Pair**: Full-duplex transmission medium simulation

### 2. MAC Layer (`mac/`)

- **Frame Format**: Preamble + Dst MAC + Src MAC + Type + Payload + CRC
- **CSMA/CD**: Carrier sense and collision detection (for shared medium)
- **L2 Switch**: MAC address learning and forwarding table

### 3. Transport Layer (`tcp/`)

- **Go-Back-N ARQ**: Sliding window with cumulative ACK
- **Selective Repeat ARQ**: Per-packet ACK with out-of-order buffering
- **Session Management**: 4-tuple connection identification
- **Socket API**: `bind()`, `listen()`, `accept()`, `connect()`, `send()`, `recv()`

### 4. Application Layer (`app/`)

- **HTTP Client**: Non-blocking GET/POST requests with callbacks
- **HTTP Server**: Route-based request handling
- **Message Parsing**: HTTP/1.1 request/response parsing

## 🚀 Quick Start

### Basic Example: Two Nodes Communication

```python
from core import PhySimulationEngine
from phy import PhyLayer, TwistedPair, Cable
from mac import MacLayer, Switcher
from tcp import TransportLayer_GBN, socket

# Create simulation engine
simulator = PhySimulationEngine(time_step_us=1)

# Create nodes with full protocol stack
class Node:
    def __init__(self, simulator, mac_addr, name):
        self.phy = PhyLayer(lower_layer=None, coding=True, 
                           simulator=simulator, name=name)
        self.mac = MacLayer(lower_layer=self.phy, simulator=simulator,
                           mode='node', mac_addr=mac_addr, name=name)
        self.tcp = TransportLayer_GBN(lower_layer=self.mac, 
                                      simulator=simulator, name=name)
        self.socket = socket(tcp_layer=self.tcp)
        self.socket.bind(8080)

node1 = Node(simulator, mac_addr=1, name='node1')
node2 = Node(simulator, mac_addr=2, name='node2')

# Create network infrastructure
cable = Cable(length=100, attenuation=3, noise_level=4)
tp = TwistedPair(cable=cable, simulator=simulator, ID=0)

# Connect nodes
tp.connect(tx_interface=node1.phy.tx_entity, rx_interface=node1.phy.rx_entity)
tp.connect(tx_interface=node2.phy.tx_entity, rx_interface=node2.phy.rx_entity)

# Run simulation
simulator.run(duration_ticks=10000)
```

### HTTP Communication Example

```python
from app.server import HttpServer
from app.client import HttpClient

# Create HTTP server and client
server = HttpServer(simulator=simulator, mac_addr=1, name='server', port=80)
client = HttpClient(simulator=simulator, mac_addr=2, name='client')

# Add custom route
server.add_route('/api/data', lambda req: b'{"status": "ok"}')

# Send HTTP request (non-blocking)
def on_response(resp):
    print(f"Received: {resp['status_code']} {resp['body']}")

client.get(dst_mac=1, dst_port=80, path='/api/data', callback=on_response)

# Run simulation
simulator.run(duration_ticks=10000)
```

## 🧪 Running Tests

```bash
# Test full protocol stack
cd test
python test_stack.py

# Test HTTP layer
cd app
python test_http.py
```

## 📊 Configuration Parameters

| Component | Parameter | Default | Description |
|-----------|-----------|---------|-------------|
| Cable | `length` | 100m | Cable length |
| Cable | `attenuation` | 3 dB | Signal attenuation |
| Cable | `noise_level` | 4 | Gaussian noise σ |
| TCP | `TIMEOUT_TICKS` | 128 | Retransmission timeout |
| TCP | `WINDOW_SIZE` | 12 | Sliding window size |
| TCP | `MSS` | 1024 | Maximum segment size |
| Simulator | `time_step_us` | 1.0 | Microseconds per tick |

## 🎯 Design Principles

1. **Separation of Concerns**: Each protocol layer handles its own responsibilities
2. **Non-blocking I/O**: All socket operations are non-blocking for simulation compatibility
3. **Realistic Timing**: Physical propagation delays and protocol timeouts are accurately modeled
4. **Extensibility**: Easy to add new protocol layers or modify existing ones

## 📝 API Reference

### Socket API

```python
sock = socket(tcp_layer)
sock.bind(port)              # Bind to local port
sock.listen(backlog)         # Start listening (server)
sock.accept() -> socket      # Accept connection (non-blocking)
sock.connect(mac, port)      # Connect to remote (client)
sock.send(data: bytes)       # Send data
sock.recv(length) -> bytes   # Receive data (non-blocking)
```

### SimulationEntity Interface

```python
class MyEntity(SimulationEntity):
    def __init__(self, name):
        super().__init__(name)
    
    def update(self, tick):
        # Called every simulation tick
        super().update(tick)
        # Your logic here
    
    def reset(self):
        # Called when simulation resets
        super().reset()
```

## 🔮 Future Work

- [ ] Event-based visualization (sequence diagrams)
- [ ] Congestion control (TCP Tahoe/Reno)
- [ ] IP layer with routing
- [ ] Wireless channel simulation
- [ ] Real-time GUI visualization

## 📄 License

MIT License

## 👤 Author

Apollo - Network Communication Project