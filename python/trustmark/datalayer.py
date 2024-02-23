# Copyright 2023 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import numpy as np 
from typing import List, Tuple
from copy import deepcopy

BCH_POLYNOMIAL = 137

class DataLayer(object):
    def __init__(self, payload_len, verbose=True, encoding_mode=0, **kw_args):

        self.bch_encoder=self.buildBCH(encoding_mode)
        self.encoding_mode=encoding_mode
        self.versionbits=4

        self.bch_decoders=dict()
        for i in range(0,5):
          self.bch_decoders[i]=self.buildBCH(i)
        self.payload_len = payload_len  # in bits

    def schemaInfo(self, version):
        if version==0:
            return 'Default'
        if version==1:
            return 'BCH_5'
        if version==2:
            return 'BCH_4'
        if version==3:
            return 'BCH_3'
        if version==4:
            return 'BCH_2'
        return 'Unknown'

    def schemaCapacity(self, version):
        if version==0:
            return 56
        if version==1:
            return 61
        if version==2:
            return 68
        if version==3:
            return 75
        if version==4:
            return 82
        return 0


    def buildBCH(self, encoding_mode):
        if encoding_mode==1:
             return (BCH(5,BCH_POLYNOMIAL))
        elif encoding_mode==2:
             return (BCH(4,BCH_POLYNOMIAL))
        elif encoding_mode==3:
             return (BCH(3,BCH_POLYNOMIAL))
        elif encoding_mode==4:
             return (BCH(2,BCH_POLYNOMIAL))
        else:  # assume default/mode 0
             return(BCH(5,BCH_POLYNOMIAL))




    def raw_payload_split(self, packet):

        packet = ''.join([str(int(bit)) for bit in packet])  # bit string

        wm_version = int(packet[-(self.versionbits):],2) # from last 4 bits
#        print('Found watermark with encoding schema %s' % self.schemaInfo(wm_version))

        if wm_version==0:
                # Default operation, 56 bit payload 5 bitflips/35 ecc bits
                packet = packet[:(len(packet)//8*8)] # trim to multiple of 8 bits
                packet = bytes(int(packet[i: i + 8], 2) for i in range(0, len(packet), 8))
                ecc_bytes=5
                data, ecc = packet[:-ecc_bytes], packet[-ecc_bytes:]
                data = ''.join(format(x, '08b') for x in data)
                ecc = ''.join(format(x, '08b') for x in ecc)
                bitflips=5
                decoder=self.bch_decoders[0]
        elif wm_version==1:
                # 5 bitflips via 35 ecc bits, 61 bit payload
                data=packet[0:61]
                ecc=packet[61:96]
                bitflips=5
                decoder=self.bch_decoders[1]
        elif wm_version==2:
                # 4 bitflips via 28 ecc bits, 68 bit payload
                data=packet[0:68]
                ecc=packet[68:96]
                bitflips=4
                decoder=self.bch_decoders[2]
        elif wm_version==3:
                # 3 bitflips via 21 ecc bits, 75 bit payload
                data=packet[0:75]
                ecc=packet[75:96]
                bitflips=3
                decoder=self.bch_decoders[3]
        elif wm_version==4:
                # 2 bitflips via 14 ecc bits, 82 bit payload
                data=packet[0:82]
                ecc=packet[82:96]
                bitflips=2
                decoder=self.bch_decoders[4]
        else:
                data=''
                ecc=''
                bitflips=-1
                decoder=None

        if decoder:
                bitflips=decoder.ECCstate.t

        return (bitflips,data,ecc,decoder,wm_version)  # unsupported or corrupt wmark



    def encode_text(self, text: List[str]):
        return np.array([self._encode_text(t) for t in text])

    def encode_binary(self, text: List[str]):
        return np.array([self._encode_binary(t) for t in text])

    def _encode_binary(self, strbin):
        return self.process_encode(str(strbin))

    def _encode_text(self, text: str):
        data = self.encode_text_ascii(text)  # bytearray
        packet_d = ''.join(format(x, '08b') for x in data)
        return self.process_encode(packet_d)

    def process_encode(self,packet_d):
        data_bitcount=self.payload_len-self.bch_encoder.get_ecc_bits()-self.versionbits
        if (self.encoding_mode==0):
            data_bitcount=56
        ecc_bitcount=self.bch_encoder.get_ecc_bits()

        packet_d=packet_d[0:data_bitcount]
        packet_d = packet_d+'0'*(data_bitcount-len(packet_d))

        if (len(packet_d)%8)==0:
           pad_d=0
        else:
           pad_d=8-len(packet_d)% 8
        paddedpacket_d = packet_d + ('0'*pad_d)
        padded_data = bytearray(bytes(int(paddedpacket_d[i: i + 8], 2) for i in range(0, len(paddedpacket_d), 8)))
 
        ecc = self.bch_encoder.encode(padded_data)  

        packet_e = ''.join(format(x, '08b') for x in ecc)
        packet_e = packet_e[0:ecc_bitcount]
        if (len(packet_e)%8)==0 or not (self.encoding_mode==0):
           pad_e=0
        else:
           pad_e=8-len(packet_e)% 8
        packet_e = packet_e + ('0'*pad_e)

        version=self.encoding_mode
        packet_v = ''.join(format(version, '04b'))
        packet = packet_d + packet_e + packet_v
        packet = [int(x) for x in packet]
        assert self.payload_len == len(packet),f'Error! Could not form complete packet'
        packet = np.array(packet, dtype=np.float32)
        return packet
    
    def decode_bitstream(self, data: np.array, MODE='text'):
        assert len(data.shape)==2
        return [self._decode_text(d, MODE) for d in data]



    def decode_text(self, data: np.array):
        assert len(data.shape)==2
        return [self._decode_text(d) for d in data]


    def decode_binary(self, data: np.array):
        assert len(data.shape)==2
        return [self._decode_text(d) for d in data]

    
    def _decode_text(self, packet: np.array, MODE):
        assert len(packet.shape)==1
        bitflips, packet_d, packet_e, bch_decoder, version = self.raw_payload_split(packet)
        if (bitflips==-1): # unsupported or corrupt wm
            return '', False, version
        if (len(packet_d)%8 ==0):
           pad_d=0
        else:
           pad_d=8-len(packet_d)% 8
        if (len(packet_e)%8 ==0):
           pad_e=0
        else:
           pad_e=8-len(packet_e)% 8
        packet_d = packet_d + ('0'*pad_d)
        packet_e = packet_e + ('0'*pad_e)

        packet_d = bytes(int(packet_d[i: i + 8], 2) for i in range(0, len(packet_d), 8))
        packet_e = bytes(int(packet_e[i: i + 8], 2) for i in range(0, len(packet_e), 8))
        data = bytearray(packet_d)
        ecc = bytearray(packet_e)
        data0 = self.decode_text_ascii(deepcopy(data)).rstrip('\x00').strip()
        if len(ecc)==bch_decoder.get_ecc_bytes():
            bitflips = bch_decoder.decode(data, ecc) 
        else:
            bitflips = -1 
        if bitflips == -1:
            data = data0
            return data, False, 0
        else:
            if MODE=='text':
                dataasc = self.decode_text_ascii(data).rstrip('\x00').strip()
            else:
                dataasc = ''.join(format(x, '08b') for x in data)
                maxbits=self.schemaCapacity(version)
                dataasc=dataasc[0:maxbits]
            return dataasc, True, version


    def encode_text_ascii(self, text: str):
        # encode text to 7-bit ascii
        # input: text, str
        # output: encoded text, bytearray
        text_int7 = [ord(t) & 127 for t in text]
        text_bitstr = ''.join(format(t,'07b') for t in text_int7)
        if len(text_bitstr) % 8 != 0:
            text_bitstr =  text_bitstr + '0'*(8-len(text_bitstr)%8) # + text_bitstr  # pad to multiple of 8
        text_int8 = [int(text_bitstr[i:i+8], 2) for i in range(0, len(text_bitstr), 8)]
        return bytearray(text_int8)


    def decode_text_ascii(self, text: bytearray):
        # decode text from 7-bit ascii
        # input: text, bytearray
        # output: decoded text, str
        text_bitstr = ''.join(format(t,'08b') for t in text)  # bit string
        text_int7 = [int(text_bitstr[i:i+7], 2) for i in range(0, len(text_bitstr), 7)]
        text_bytes = bytes(text_int7)
        return text_bytes.decode('utf-8')



## TESTING ONLY

# Copyright 2023 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import random
import uuid


def random_string(string_length=7):
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.
  
def main():

    # test rig for error correction
    N=100
    for i in range(0,N):
       string_secret=random_string(random.randint(4, 8))
       mode=random.randint(0, 4) 
       D=DataLayer(100,False,mode)
       wmpayload = D.encode_text([string_secret])   
       bitflips= random.randint(0,D.bch_encoder.ECCstate.t)   
       CORRUPTBITS=[random.randint(0, len(wmpayload[0])-5) for _ in range(bitflips)]
       for b in CORRUPTBITS:
          wmpayload[0][b] = 1- wmpayload[0][b]
       secret_pred, detected, version = D.decode_bitstream(wmpayload,MODE='text')[0]
       print("Recovered - orig[%s] recover[%s] (det=%s schema %d bitflips=%d)" % (string_secret,secret_pred,detected,version,bitflips))
       for c in range(0,len(string_secret)): 
         if string_secret[c]!=secret_pred[c]:
          print("STOP - MISMATCH!")
          quit()

if __name__ == "__main__":
    from bchecc import BCH    
    main()
else:
    from .bchecc import BCH
 

