"""
test/utils.py

utils functions for testing
"""

import random

def generate_random_data(length:int)->bytes:
    """generate random bytes of specified length"""
    return bytes(random.getrandbits(8) for _ in range(length))

def diff(a:bytes, b:bytes)->int:
    """calculate byte-wise difference between two byte sequences"""
    diff_list:list[tuple[int,int,int]]= []
    length= min(len(a), len(b))
    diffs= 0
    for i in range(length):
        if a[i] != b[i]:
            diffs += 1
            diff_list.append((i, a[i], b[i]))
    diffs += abs(len(a)-len(b))

    for i in range(len(diff_list)):
        index, byte_a, byte_b= diff_list[i]
        byte_a= int.to_bytes(byte_a, length=1, byteorder='big')
        byte_b= int.to_bytes(byte_b, length=1, byteorder='big')
        print(f'Byte {index}: {byte_a} != {byte_b}')

    return diffs


