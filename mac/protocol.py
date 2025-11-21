"""
realize Mac Layer encoding and decoding logic

frame format:\n
[2B Header]+[2B Length]+[1B Src_Mac]+[1B Dst_Mac]+[nB Payload]+[4B CRC32]
"""

import struct
import zlib

HEADER = b"\xaa\xbb"


class NetworkInterface:
    def __init__(self, mac_addr: str, name: str = "eth0"):
        self.mac_addr = mac_addr
        self.name = name
        self.header = HEADER

    def encoding(self, dst_mac: str, data: bytes) -> bytes:
        lens = 1 + 1 + 1 + len(data)
        header = struct.pack("!2sHBB", self.header, lens, self.mac_addr, dst_mac)
        # header = struct.pack("!2sHBB", b'\xaa\xbb', 6, '1', '2')
        raw_frame = header + data
        crc = zlib.crc32(raw_frame)
        frame = raw_frame + struct.pack("!I", crc)
        return frame

    def decoding(self, frame: bytes) -> tuple[str, bytes] | None:
        if len(frame) < 2 + 2 + 1 + 1 + 4:
            return None
        if frame[:2] != self.header:
            return None
        recieved_crc: int = struct.unpack("!I", frame[-4:])[0]
        caculated_crc = zlib.crc32(frame[:-4])
        if recieved_crc != caculated_crc:
            print(
                f"[{self.name}] CRC Error: recieved {recieved_crc}, caculated {caculated_crc}"
            )
            raise ValueError("CRC Mismatch")

        _, lens, src_mac, dst_mac = struct.unpack("!2sHBB", frame[:6])

        if dst_mac != self.mac_addr:
            print(
                f"[{self.name}] MAC Address Mismatch: dst {dst_mac}, self {self.mac_addr}"
            )
            raise ValueError("MAC Address Mismatch")

        data = frame[6:-4]
        return src_mac, data


def test():
    ni_0 = NetworkInterface(mac_addr=1, name="test_eth0")
    ni_1 = NetworkInterface(mac_addr=2, name="test_eth1")

    data = b"Hello, this is a test message."
    frame: bytes = ni_0.encoding(dst_mac=2, data=data)
    print(f"Encoded Frame: {frame}")

    idx_to_corrupt = 10
    corrupted_frame = bytearray(frame)
    corrupted_frame[idx_to_corrupt] ^= 0x01  # Flip bits to corrupt
    frame = bytes(corrupted_frame)

    result = ni_1.decoding(frame)
    print(f"Decoded Result: {result}")
    assert result == (1, data)


if __name__ == "__main__":
    test()
