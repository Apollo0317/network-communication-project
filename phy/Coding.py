import sys

sys.path.append("..")
import numpy
from phy.modulator import Modulator, DeModulator


class Hamming:
    """
    Hamming (7,4) error correction code implementation.
    Can be used to encode, parity check, error correct, decode and get the orginal message back.
    This can detect two bit errors and correct single bit errors.
    """

    _gT = numpy.matrix(
        [
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 1, 1, 1],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    _h = numpy.matrix(
        [[1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1]]
    )

    _R = numpy.matrix(
        [
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ]
    )

    @staticmethod
    def _strToMat(binaryStr):
        """
        @Input
        Binary string of length 4

        @Output
        Numpy row vector of length 4
        """

        inp = numpy.frombuffer(binaryStr.encode(), dtype=numpy.uint8) - ord("0")
        return inp

    @staticmethod
    def _listToMat(bit_list: list[int]):
        return numpy.array(bit_list)

    @staticmethod
    def EncToList(array: numpy.ndarray) -> list[int]:
        bit_list = array.tolist()
        return [ele[0] for ele in bit_list]

    @staticmethod
    def ListToEnc(bit_list: list[int]) -> numpy.ndarray:
        bit_list = [[ele] for ele in bit_list]
        array = numpy.array(bit_list)
        return array

    @staticmethod
    def test_filp_bit(bit_list: list[int], pos: int):
        bit_list[pos] ^= 1

    def encode(self, message: list[int]):
        """
        @Input
        String
        Message is a 4 bit binary string
        @Output
        Numpy matrix column vector
        Encoded 7 bit binary string
        """
        message = numpy.matrix(self._listToMat(message)).transpose()
        en = numpy.dot(self._gT, message) % 2
        return self.EncToList(array=en)

    def parityCheck(self, message):
        """
        @Input
        Numpy matrix a column vector of length 7
        Accepts a binary column vector

        @Output
        Numpy row vector of length 3
        Returns the single bit error location as row vector
        """
        z = numpy.dot(self._h, message) % 2
        return numpy.fliplr(z.transpose())

    def getOriginalMessage(self, message: list[int]) -> list[int]:
        """
        @Input
        Numpy matrix a column vector of length 7
        Accepts a binary column vector

        @Output
        List of length 4
        Returns the single bit error location as row vector ()
        """
        message = self.ListToEnc(bit_list=message)
        ep = self.parityCheck(message)
        pos = self._binatodeci(ep)
        if pos > 0:
            # print(
            #     f"bit flip in {pos - 1}th bit: {message[pos - 1][0] ^ 1}->{message[pos - 1][0]}"
            # )
            correctMessage = self._flipbit(message, pos)
        else:
            correctMessage = message

        origMessage = numpy.dot(self._R, correctMessage)
        return origMessage.transpose().tolist()[0]

    def _flipbit(self, enc, bitpos):
        """
        @Input
          enc:Numpy matrix a column vector of length 7
          Accepts a binary column vector

          bitpos: Integer value of the position to change
          flip the bit. Value should be on range 1-7

        @Output
          Numpy matrix a column vector of length 7
          Returns the bit flipped matrix
        """
        enc = enc.transpose().tolist()
        bitpos = bitpos - 1
        if enc[0][bitpos] == 1:
            enc[0][bitpos] = 0
        else:
            enc[0][bitpos] = 1
        return numpy.matrix(enc).transpose()

    def _binatodeci(self, binaryList):
        """
        @Input
        Numpy matrix column or row one dimension

        @Output
        Decimal number equal to the binary matrix
        """
        return sum(
            val * (2**idx) for idx, val in enumerate(reversed(binaryList.tolist()[0]))
        )


class ChannelEncoder:
    def __init__(self):
        self.ham = Hamming()
        self.n = 7
        self.k = 4
        self.byte_length = 8

    @staticmethod
    def bytes_to_bits(byte_data: bytes) -> list[int]:
        return Modulator.bytes_to_bits(byte_data=byte_data)

    @staticmethod
    def bits_to_bytes(bits: list[int]) -> bytes:
        return DeModulator.bits_to_bytes(bit_list=bits)

    def _encoding(self, bit_list: list[int]) -> list[int]:
        # bit_list= self.bytes_to_bits(data)
        length = len(bit_list)
        remain_bit = length % self.k
        # Zero padding to align to k-bit boundaries
        if remain_bit:
            zero_pad = [0] * (self.k - remain_bit)
            #print(f"pad zero during enc: {len(zero_pad)}")
            bit_list.extend(zero_pad)
            # Group bits into chunks of 4
        grouped_bit_list = []
        for i in range(0, len(bit_list), self.k):
            grouped_bit_list.append(
                [bit_list[i], bit_list[i + 1], bit_list[i + 2], bit_list[i + 3]]
            )
        encoded_bit_list = []
        for bit_group in grouped_bit_list:
            encoded_bit_list.extend(self.ham.encode(message=bit_group))
        return encoded_bit_list

    def _decoding(self, bit_list: list[int]) -> list[int]:
        length = len(bit_list)
        remain_bit = length % self.n
        # Zero padding to align to k-bit boundaries
        if remain_bit:
            zero_pad = [0] * (self.n - remain_bit)
            #print(f"pad zero during dec: {len(zero_pad)}")
            bit_list.extend(zero_pad)
            # Group bits into chunks of 4
        decoded_bit_list: list[int] = []
        for i in range(0, len(bit_list), self.n):
            cur_group: list[int] = []
            for j in range(self.n):
                cur_group.append(bit_list[i + j])
            decoded_bit_list.extend(self.ham.getOriginalMessage(message=cur_group))
        return decoded_bit_list

    def encoding(self, data: bytes) -> bytes:
        bit_list = self.bytes_to_bits(byte_data=data)
        encoded_bit_list = self._encoding(bit_list=bit_list)

        length = len(encoded_bit_list)
        remain_bit = length % self.byte_length
        # Zero padding to align to k-bit boundaries
        if remain_bit:
            zero_pad = [0] * (self.byte_length - remain_bit)
            #print(f"pad zero during enc_1: {len(zero_pad)}")
            encoded_bit_list.extend(zero_pad)
        return self.bits_to_bytes(bits=encoded_bit_list)

    def decoding(self, data: bytes) -> bytes:
        bit_list = self.bytes_to_bits(byte_data=data)
        decoded_bit_list = self._decoding(bit_list=bit_list)
        return self.bits_to_bytes(bits=decoded_bit_list)


if __name__ == "__main__":
    channel_encoder = ChannelEncoder()

    # 更全面的测试用例
    test_cases = [
        # 基础长度测试
        (b"1", "1字节"),
        (b"12", "2字节"),
        (b"123", "3字节"),
        (b"1234", "4字节(恰好对齐)"),
        (b"12345", "5字节"),
        (b"123456", "6字节"),
        (b"1234567", "7字节"),
        (b"12345678", "8字节"),
        (b"123454321", "9字节"),
        # 边界情况
        (b"", "空字节"),
        (b"a" * 100, "100字节"),
        (b"x" * 255, "255字节"),
        (b"y" * 256, "256字节"),
        # 特殊字符
        (b"\x00", "null字节"),
        (b"\x00\x00\x00", "多个null字节"),
        (b"\xff", "0xFF字节"),
        (b"\x00\xff\x00", "混合字节"),
        # Unicode测试
        (bytes("你好".encode("utf-8")), "中文UTF-8"),
        (bytes("Hello World!".encode("utf-8")), "英文UTF-8"),
        (bytes("🎉🎊".encode("utf-8")), "Emoji UTF-8"),
        # 二进制数据
        (bytes(range(16)), "0-15连续字节"),
        (bytes([0b10101010] * 10), "交替bit模式"),
        (bytes([0b11111111, 0b00000000] * 5), "全1全0交替"),
    ]

    print("=" * 60)
    print("开始测试...")
    print("=" * 60)

    passed = 0
    failed = 0

    for test_data, description in test_cases:
        try:
            print(f"\n测试: {description}")
            print(f"原始: 长度={len(test_data)}, 前10字节={test_data[:10]}")

            enc = channel_encoder.encoding(data=test_data)
            print(f"编码: 长度={len(enc)}")

            dec = channel_encoder.decoding(data=enc)
            print(f"解码: 长度={len(dec)}, 前10字节={dec[:10]}")

            # 严格检查
            length_match = len(test_data) == len(dec)
            content_match = test_data == dec

            print(f"长度匹配: {length_match}")
            print(f"内容匹配: {content_match}")

            if not length_match:
                print(
                    f"❌ 长度不匹配! 原始={len(test_data)}, 解码={len(dec)}, 差={len(dec) - len(test_data)}"
                )
                failed += 1
            elif not content_match:
                print(f"❌ 内容不匹配!")
                # 显示差异
                for i in range(min(len(test_data), len(dec))):
                    if test_data[i : i + 1] != dec[i : i + 1]:
                        print(f"  位置{i}: 原始={test_data[i]:02x}, 解码={dec[i]:02x}")
                        if i > 10:
                            print("  ...")
                            break
                failed += 1
            else:
                print(f"✓ 通过")
                passed += 1

        except Exception as e:
            print(f"❌ 异常: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"测试结果: 通过 {passed}/{len(test_cases)}, 失败 {failed}/{len(test_cases)}")
    print("=" * 60)

    if failed > 0:
        print("\n⚠️  存在失败的测试用例,建议添加长度记录机制!")
    else:
        print("\n✓ 所有测试通过!")
