"""
Modulator and DeModulator for 16-QAM scheme

Provides phy layer interface
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import json
import time
from cable import Cable

class Modulator:
    def __init__(self, scheme:str, symbol_rate:int, sample_rate:int, fc:int):
        self.scheme= scheme
        self.symbol_rate= symbol_rate
        self.sample_rate= sample_rate
        self.fc= fc
        self.mapping= Modulator.generate_QAM_mapping()
        pass
    
    @staticmethod
    def bytes_to_bits(byte_data: bytes)->list:
        bit_list=[]
        for byte in byte_data:
            bit_list.extend([(byte>>i) & 1 for i in range(7,-1,-1)])
        return bit_list

    @staticmethod
    def generate_QAM_mapping(order:int=16)->dict[str,list]:
        mapping={}
        for code in range(order):
            bit_list= [(code>>i) & 1 for i in range(3,-1,-1)]
            I_bit, Q_bit= bit_list[:2], bit_list[2:]
            if I_bit==[0,0]:
                I_out=-3
            elif I_bit==[0,1]:
                I_out=-1
            elif I_bit==[1,1]:
                I_out=1
            else:
                I_out=3
            if Q_bit==[0,0]:
                Q_out=-3
            elif Q_bit==[0,1]:
                Q_out=-1
            elif Q_bit==[1,1]:
                Q_out=1
            else:
                Q_out=3
            mapping[str(bit_list)]=[I_out,Q_out]
        return mapping
    
    def Set_Frequency(self, fc:int):
        if fc<0:
            raise  ValueError('invalid frequency')
        self.fc= fc
        print(f'fc set to {self.fc}Hz now')
            
    def QAM(self, byte_data:bytes)->list:
        bit_list= Modulator.bytes_to_bits(byte_data)
        bit_per_symbol= 4
        length=len(bit_list)
        remain_bit=length % bit_per_symbol
        if remain_bit:
            zero_pad= [0]*(bit_per_symbol-remain_bit)
            print(f'pad zero: {len(zero_pad)}')
            bit_list.extend(zero_pad)
        grouped_bit_list=[]
        for i in range(0, len(bit_list), bit_per_symbol):
            grouped_bit_list.append([bit_list[i],bit_list[i+1],bit_list[i+2],bit_list[i+3]])
        #print(grouped_bit_list)
        symbols= [self.mapping.get(str(bit_group)) for bit_group in grouped_bit_list]
        return symbols
    
    def QAM_UpConverter(self, symbols:list[list], debug=False)->np.ndarray:
        symbols:list[complex]= [complex(symbol[0],symbol[1]) for symbol in symbols]
        complex_symbols= np.array(symbols, dtype=np.complex128)
        #sps:sample per symbol
        sps= self.sample_rate / self.symbol_rate
        I_baseband= np.real(complex_symbols)
        Q_baseband= np.imag(complex_symbols)

        #pulse shaping: recangle filter only now
        #TODO add RRC pulse shaping filter
        I_n= np.repeat(I_baseband, repeats=sps)
        Q_n= np.repeat(Q_baseband, repeats=sps)

        lens= len(I_n)
        n=np.arange(lens)
        time= n/self.sample_rate
        time= time*1e6
        carrier_cos= np.cos(2*np.pi*self.fc*n/self.sample_rate)
        carrier_sin= np.sin(2*np.pi*self.fc*n/self.sample_rate)

        I_modulated= I_n*carrier_cos
        Q_modulated= Q_n* (-carrier_sin)

        qam_signal= I_modulated + Q_modulated

        if debug:
            plt.figure(figsize=(12,6))

            plt.subplot(3,2,1)
            plt.plot(time, carrier_cos)
            plt.title('cos carrier')
            plt.xlabel('t/us')
            plt.ylabel('V/volt')
            plt.grid(True)

            plt.subplot(3,2,2)
            plt.plot(time, carrier_sin)
            plt.title('sin carrier')
            plt.xlabel('t/us')
            plt.ylabel('V/volt')
            plt.grid(True)

            plt.subplot(3,2,3)
            plt.plot(time, I_n)
            plt.title('I(t)')
            plt.xlabel('t/us')
            plt.ylabel('V/volt')
            plt.grid(True)

            plt.subplot(3,2,4)
            plt.plot(time, Q_n)
            plt.title('Q(t)')
            plt.xlabel('t/us')
            plt.ylabel('V/volt')
            plt.grid(True)

            plt.subplot(3,2,5)
            plt.plot(time, qam_signal)
            plt.title('QAM(t)')
            plt.xlabel('t/us')
            plt.ylabel('V/volt')
            plt.grid(True)

            plt.tight_layout()
            #plt.show()
            plt.savefig('fig/signal.png')

        return qam_signal

        pass

    def modulate(self, data:bytes)->np.ndarray:
        symbols= self.QAM(byte_data=data)
        #print(f'symbols: {symbols}')
        signal= self.QAM_UpConverter(symbols=symbols, debug=False)
        return signal

class DeModulator:
    def __init__(self, scheme:str, symbol_rate:int, sample_rate:int, fc:int):
        self.scheme= scheme
        self.rs= symbol_rate
        self.fs= sample_rate
        self.fc= fc
        self.sps= int(sample_rate / symbol_rate)
        self.mapping:dict[str,list]= DeModulator.generate_QAM_mapping()
        pass
    
    def generate_QAM_mapping(order:int=16)->dict[str,list]:
        pos_mapping= Modulator.generate_QAM_mapping()
        rev_mapping= {}
        for k,v in pos_mapping.items():
            bit_group:list[int]= json.loads(k)
            rev_mapping[str(v)]= bit_group
        return rev_mapping
    
    @staticmethod
    def bits_to_bytes(bit_list:list[int])->bytes:
        bit_num=len(bit_list)
        byte_list=bytearray()
        byte_num= int(bit_num/8)
        for i in range(0, byte_num):
            bits_of_byte= bit_list[i*8:(i+1)*8]
            value=0
            for j in range(8):
                value+=bits_of_byte[j]<<(7-j)
            byte_list.append(value)
        return bytes(byte_list)

        pass

    @staticmethod
    def distance(a:list[float,2], b:list[float,2])->float:
        return (a[0]-b[0])**2+(a[1]-b[1])**2

    def Detect_Symbol(self, symbols:list[list])->list[int]:
        bit_list=[]
        symbol_list:list[str]=self.mapping.keys()
        symbol_list:list[list]= [json.loads(symbol) for symbol in symbol_list]
        for symbol in symbols:
            idx= np.argmin([self.distance(symbol, symbol_i) for symbol_i in symbol_list])
            judged_symbol= str(symbol_list[idx])
            bit_list.extend(self.mapping.get(judged_symbol))
        return bit_list
        pass

    def QAM_DownConverter(self, qam_signal:np.ndarray, debug=True)->list[list]:
        lens= len(qam_signal)
        n=np.arange(lens)
        time= n/self.fs
        time_us= time*1e6
        carrier_cos= np.cos(2*np.pi*self.fc*n/self.fs)
        carrier_sin= np.sin(2*np.pi*self.fc*n/self.fs)
        I_prime= qam_signal*carrier_cos
        Q_prime= qam_signal*carrier_sin

        order = 6
        f_cutoff = 2 * self.rs
        Wn = f_cutoff / (self.fs / 2)

        b, a = signal.butter(order, Wn, 'low', analog=False)
        
        I_baseband:list[float] = list(signal.filtfilt(b, a, I_prime)*2)
        Q_baseband:list[float] = list(signal.filtfilt(b, a, Q_prime)*(-2))

        if debug:
            plt.figure(figsize=(12,8))
            plt.subplot(2,1,1)
            plt.plot(time_us, I_baseband)
            plt.title('I(t)')
            plt.grid(True)

            plt.subplot(2,1,2)
            plt.plot(time_us, Q_baseband)
            plt.title('Q(t)')
            plt.grid(True)

            plt.savefig('fig/IQ.png')
        
        symbol_num= int(lens/self.sps)
        I_n:list[float]= [sum(I_baseband[(i)*self.sps:(i+1)*self.sps])/self.sps for i in range(symbol_num)]
        Q_n:list[float]= [sum(Q_baseband[(i)*self.sps:(i+1)*self.sps])/self.sps for i in range(symbol_num)]

        symbols= [[I,Q] for (I,Q) in zip(I_n, Q_n)]

        #print(symbols)
        
        return symbols

        pass

    def demodulate(self, signal:np.ndarray)->bytes:
        symbols= self.QAM_DownConverter(qam_signal=signal, debug=False)
        bits= self.Detect_Symbol(symbols=symbols)
        byte_recovered= self.bits_to_bytes(bit_list=bits)
        return byte_recovered



def test():
    modulator=Modulator(scheme='16QAM', symbol_rate=1e6, sample_rate=50e6, fc=2e6)
    demodulator= DeModulator(scheme='16QAM', symbol_rate=1e6, sample_rate=50e6, fc=2e6)
    test_str=b'Hello, this is a test string for QAM modulation and demodulation over a simulated cable channel. '
    start_time= time.time()
    qam_signal= modulator.modulate(data=test_str)
        # Create cable (with debug mode enabled)
    cable = Cable(
        length=100,           # 100 meters
        attenuation=0.1,      # Attenuation coefficient
        noise_level=0.8,     # Noise level
        debug_mode=False      # Set to True to see waveforms
    )
    
    print(f"\n{cable}")

    recv_signal= cable.transmit(signal=qam_signal)
    #print(demodulator.mapping)
    recv_str= demodulator.demodulate(signal=recv_signal)
    cost= time.time()-start_time
    print(f'orig:{test_str[:64]}...')
    print(f'recv:{recv_str[:64]}...')
    print(f'cost:{cost}s')
    pass

if __name__=='__main__':
    test()