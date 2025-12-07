"""
realize Mac Layer encoding and decoding logic

frame format:\n
[2B Header]+[2B Length]+[1B Src_Mac]+[1B Dst_Mac]+[nB Payload]+[4B CRC32]
"""


# CRC-32 标准多项式 (IEEE 802.3)
CRC32_POLYNOMIAL = 0xEDB88320


def crc32(data: bytes) -> int:
    """
    计算 CRC-32 校验和 (IEEE 802.3 标准)
    
    Args:
        data: 待校验的字节数据
    
    Returns:
        32位 CRC 校验值 (无符号整数)
    """
    # 初始化 CRC 寄存器为全 1
    crc = 0xFFFFFFFF
    
    for byte in data:
        # 将当前字节与 CRC 低 8 位异或
        crc ^= byte
        
        # 处理 8 个 bit
        for _ in range(8):
            # 如果最低位为 1，右移并与多项式异或
            if crc & 1:
                crc = (crc >> 1) ^ CRC32_POLYNOMIAL
            else:
                crc >>= 1
    
    # 最终取反（CRC-32 标准要求）
    return crc ^ 0xFFFFFFFF


def crc32_with_table(data: bytes) -> int:
    """
    使用预计算查找表的 CRC-32 实现
    
    Args:
        data: 待校验的字节数据
    
    Returns:
        32位 CRC 校验值
    """
    # 生成查找表（只在首次调用时生成）
    if not hasattr(crc32_with_table, '_table'):
        table = []
        for i in range(256):
            crc = i
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ CRC32_POLYNOMIAL
                else:
                    crc >>= 1
            table.append(crc)
        crc32_with_table._table = table
    
    table = crc32_with_table._table
    crc = 0xFFFFFFFF
    
    for byte in data:
        crc = table[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    
    return crc ^ 0xFFFFFFFF





# ============================================================================
# 单元测试
# ============================================================================

def test_crc32_basic():
    """测试 CRC-32 基本功能"""
    import zlib
    
    test_cases = [
        b"",
        b"a",
        b"abc",
        b"Hello, World!",
        b"123456789",  # 标准测试向量，CRC-32 应为 0xCBF43926
        b"\x00" * 100,
        b"\xff" * 100,
        bytes(range(256)),
    ]
    
    print("=" * 60)
    print("CRC-32 Basic Tests")
    print("=" * 60)
    
    all_passed = True
    for data in test_cases:
        our_crc = crc32(data)
        zlib_crc = zlib.crc32(data) & 0xFFFFFFFF  # 确保无符号
        
        status = "✓" if our_crc == zlib_crc else "✗"
        if our_crc != zlib_crc:
            all_passed = False
        
        preview = data[:20].hex() + ("..." if len(data) > 20 else "")
        print(f"{status} data={preview:<45} our={our_crc:#010x} zlib={zlib_crc:#010x}")
    
    assert all_passed, "CRC-32 basic tests failed!"
    print("\n✓ All basic CRC-32 tests passed!\n")


def test_crc32_table_version():
    """测试查表法 CRC-32 与基本版本一致"""
    import zlib
    
    test_cases = [
        b"Hello, World!",
        b"123456789",
        bytes(range(256)),
        b"The quick brown fox jumps over the lazy dog",
    ]
    
    print("=" * 60)
    print("CRC-32 Table Version Tests")
    print("=" * 60)
    
    all_passed = True
    for data in test_cases:
        basic_crc = crc32(data)
        table_crc = crc32_with_table(data)
        zlib_crc = zlib.crc32(data) & 0xFFFFFFFF
        
        match = (basic_crc == table_crc == zlib_crc)
        status = "✓" if match else "✗"
        if not match:
            all_passed = False
        
        print(f"{status} basic={basic_crc:#010x} table={table_crc:#010x} zlib={zlib_crc:#010x}")
    
    assert all_passed, "CRC-32 table version tests failed!"
    print("\n✓ All table version tests passed!\n")


def test_crc32_standard_vector():
    """测试标准测试向量"""
    print("=" * 60)
    print("CRC-32 Standard Test Vector")
    print("=" * 60)
    
    # "123456789" 的 CRC-32 标准值
    data = b"123456789"
    expected = 0xCBF43926
    
    result = crc32(data)
    status = "✓" if result == expected else "✗"
    
    print(f'{status} CRC-32("123456789") = {result:#010x} (expected {expected:#010x})')
    
    assert result == expected, f"Standard vector test failed: got {result:#010x}, expected {expected:#010x}"
    print("\n✓ Standard test vector passed!\n")


def test_crc32_error_detection():
    """测试 CRC-32 错误检测能力"""
    print("=" * 60)
    print("CRC-32 Error Detection Tests")
    print("=" * 60)
    
    original_data = b"Hello, this is a test message for CRC verification."
    original_crc = crc32(original_data)
    
    errors_detected = 0
    total_tests = 0
    
    # 测试单比特翻转
    for byte_idx in range(len(original_data)):
        for bit_idx in range(8):
            corrupted = bytearray(original_data)
            corrupted[byte_idx] ^= (1 << bit_idx)
            corrupted_crc = crc32(bytes(corrupted))
            
            total_tests += 1
            if corrupted_crc != original_crc:
                errors_detected += 1
    
    detection_rate = errors_detected / total_tests * 100
    print(f"Single-bit errors: {errors_detected}/{total_tests} detected ({detection_rate:.1f}%)")
    
    # 测试双比特翻转
    errors_detected_2bit = 0
    total_2bit = 0
    
    for i in range(min(50, len(original_data))):
        for j in range(i + 1, min(50, len(original_data))):
            corrupted = bytearray(original_data)
            corrupted[i] ^= 0x01
            corrupted[j] ^= 0x01
            corrupted_crc = crc32(bytes(corrupted))
            
            total_2bit += 1
            if corrupted_crc != original_crc:
                errors_detected_2bit += 1
    
    detection_rate_2bit = errors_detected_2bit / total_2bit * 100 if total_2bit > 0 else 0
    print(f"Double-bit errors: {errors_detected_2bit}/{total_2bit} detected ({detection_rate_2bit:.1f}%)")
    
    # 测试突发错误
    errors_detected_burst = 0
    total_burst = 0
    
    for start in range(len(original_data) - 4):
        for burst_len in range(1, 5):
            corrupted = bytearray(original_data)
            for k in range(burst_len):
                if start + k < len(corrupted):
                    corrupted[start + k] ^= 0xFF
            corrupted_crc = crc32(bytes(corrupted))
            
            total_burst += 1
            if corrupted_crc != original_crc:
                errors_detected_burst += 1
    
    detection_rate_burst = errors_detected_burst / total_burst * 100 if total_burst > 0 else 0
    print(f"Burst errors: {errors_detected_burst}/{total_burst} detected ({detection_rate_burst:.1f}%)")
    
    assert detection_rate == 100, "Should detect all single-bit errors!"
    print("\n✓ Error detection tests passed!\n")




def test_performance():
    """性能测试"""
    import time
    
    print("=" * 60)
    print("CRC-32 Performance Test")
    print("=" * 60)
    
    # 生成测试数据
    data_1kb = bytes(range(256)) * 4
    data_1mb = data_1kb * 1024
    
    iterations = 1000
    
    # 测试基本版本
    start = time.perf_counter()
    for _ in range(iterations):
        crc32(data_1kb)
    basic_1kb_time = (time.perf_counter() - start) / iterations * 1000
    
    # 测试查表版本
    start = time.perf_counter()
    for _ in range(iterations):
        crc32_with_table(data_1kb)
    table_1kb_time = (time.perf_counter() - start) / iterations * 1000
    
    print(f"1KB data ({iterations} iterations):")
    print(f"  Basic version: {basic_1kb_time:.4f} ms/call")
    print(f"  Table version: {table_1kb_time:.4f} ms/call")
    print(f"  Speedup: {basic_1kb_time / table_1kb_time:.2f}x")
    
    iterations_large = 10
    
    start = time.perf_counter()
    for _ in range(iterations_large):
        crc32_with_table(data_1mb)
    table_1mb_time = (time.perf_counter() - start) / iterations_large * 1000
    
    throughput = len(data_1mb) / (table_1mb_time / 1000) / (1024 * 1024)
    print(f"\n1MB data (table version):")
    print(f"  Time: {table_1mb_time:.2f} ms")
    print(f"  Throughput: {throughput:.1f} MB/s")
    
    print("\n✓ Performance test completed!\n")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Running All CRC-32 Unit Tests")
    print("=" * 60 + "\n")
    
    test_crc32_basic()
    test_crc32_table_version()
    test_crc32_standard_vector()
    test_crc32_error_detection()
    test_performance()
    


if __name__ == "__main__":
    run_all_tests()