/*
 Copyright 2022 Adobe
 All Rights Reserved.

 NOTICE: Adobe permits you to use, modify, and distribute this file in
 accordance with the terms of the Adobe license agreement accompanying
 it. 
*/
#include "bch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    struct bch_control *bch;
    uint8_t *ecc;
    unsigned int data_len;
    unsigned int *errloc;
    int nerr;
} BCHObject;

unsigned char reverse_byte(unsigned char x)
{
    static const unsigned char table[] = {
        0x00, 0x80, 0x40, 0xc0, 0x20, 0xa0, 0x60, 0xe0,
        0x10, 0x90, 0x50, 0xd0, 0x30, 0xb0, 0x70, 0xf0,
        0x08, 0x88, 0x48, 0xc8, 0x28, 0xa8, 0x68, 0xe8,
        0x18, 0x98, 0x58, 0xd8, 0x38, 0xb8, 0x78, 0xf8,
        0x04, 0x84, 0x44, 0xc4, 0x24, 0xa4, 0x64, 0xe4,
        0x14, 0x94, 0x54, 0xd4, 0x34, 0xb4, 0x74, 0xf4,
        0x0c, 0x8c, 0x4c, 0xcc, 0x2c, 0xac, 0x6c, 0xec,
        0x1c, 0x9c, 0x5c, 0xdc, 0x3c, 0xbc, 0x7c, 0xfc,
        0x02, 0x82, 0x42, 0xc2, 0x22, 0xa2, 0x62, 0xe2,
        0x12, 0x92, 0x52, 0xd2, 0x32, 0xb2, 0x72, 0xf2,
        0x0a, 0x8a, 0x4a, 0xca, 0x2a, 0xaa, 0x6a, 0xea,
        0x1a, 0x9a, 0x5a, 0xda, 0x3a, 0xba, 0x7a, 0xfa,
        0x06, 0x86, 0x46, 0xc6, 0x26, 0xa6, 0x66, 0xe6,
        0x16, 0x96, 0x56, 0xd6, 0x36, 0xb6, 0x76, 0xf6,
        0x0e, 0x8e, 0x4e, 0xce, 0x2e, 0xae, 0x6e, 0xee,
        0x1e, 0x9e, 0x5e, 0xde, 0x3e, 0xbe, 0x7e, 0xfe,
        0x01, 0x81, 0x41, 0xc1, 0x21, 0xa1, 0x61, 0xe1,
        0x11, 0x91, 0x51, 0xd1, 0x31, 0xb1, 0x71, 0xf1,
        0x09, 0x89, 0x49, 0xc9, 0x29, 0xa9, 0x69, 0xe9,
        0x19, 0x99, 0x59, 0xd9, 0x39, 0xb9, 0x79, 0xf9,
        0x05, 0x85, 0x45, 0xc5, 0x25, 0xa5, 0x65, 0xe5,
        0x15, 0x95, 0x55, 0xd5, 0x35, 0xb5, 0x75, 0xf5,
        0x0d, 0x8d, 0x4d, 0xcd, 0x2d, 0xad, 0x6d, 0xed,
        0x1d, 0x9d, 0x5d, 0xdd, 0x3d, 0xbd, 0x7d, 0xfd,
        0x03, 0x83, 0x43, 0xc3, 0x23, 0xa3, 0x63, 0xe3,
        0x13, 0x93, 0x53, 0xd3, 0x33, 0xb3, 0x73, 0xf3,
        0x0b, 0x8b, 0x4b, 0xcb, 0x2b, 0xab, 0x6b, 0xeb,
        0x1b, 0x9b, 0x5b, 0xdb, 0x3b, 0xbb, 0x7b, 0xfb,
        0x07, 0x87, 0x47, 0xc7, 0x27, 0xa7, 0x67, 0xe7,
        0x17, 0x97, 0x57, 0xd7, 0x37, 0xb7, 0x77, 0xf7,
        0x0f, 0x8f, 0x4f, 0xcf, 0x2f, 0xaf, 0x6f, 0xef,
        0x1f, 0x9f, 0x5f, 0xdf, 0x3f, 0xbf, 0x7f, 0xff,
    };
    return table[x];
}

int BCH_alloc(BCHObject *self, int m, int t, unsigned int prim_poly, int swap_bits)
{

    printf("BCH params: m=%d, t=%d, primpoly=%d, swapbits=%d\n",m,t,prim_poly,swap_bits);
    self->bch = bch_init(m, t, prim_poly, swap_bits);
    if (!self->bch) {
        printf("Failed to alloc bch\n");
        return 0;
    }

            
    self->ecc = calloc(1, self->bch->ecc_bytes);
    if (!self->ecc) {
        bch_free(self->bch);
        self->bch = NULL;
        printf("Failed to alloc ecc\n");
        return 0;
    }
     
    self->errloc = calloc(1, sizeof(unsigned int) * self->bch->t);
    if (!self->errloc) {
        bch_free(self->bch);
        self->bch = NULL;
        free(self->ecc);   
        self->ecc = NULL;
        printf("Failed to alloc errloc\n");
        return 0;
    }
    
    memset(self->bch->syn, 0, sizeof(unsigned int) * 2*self->bch->t);
        
    return -1;

}


void BCH_dealloc(BCHObject *self)
{
    if (self->bch) {
        bch_free(self->bch);
        self->bch = NULL;
    }

    if (self->ecc) {
        free(self->ecc);
        self->ecc = NULL;
    }

    if (self->errloc) {
        free(self->errloc);
        self->errloc = NULL;
    }

}

void ascii7(unsigned char* in, unsigned char* out) {
  unsigned bit_count = 0;
  unsigned bit_queue = 0;
  while (*in) {
    bit_queue |= (reverse_byte(*in & 0x7Fu) >> 1)  << bit_count;
    bit_count += 7;
    if (bit_count >= 8) {
      *out++ = reverse_byte((unsigned char) bit_queue);
      bit_count -= 8;
      bit_queue >>= 8;
    }
    in++;
  }
  if (bit_count > 0) {
    *out++ = reverse_byte((unsigned char) bit_queue);
    }
}


unsigned int convertTo7BitAscii(unsigned char* data, unsigned int datalen, unsigned char** out) {

    int bytes= (datalen *7)/8;
    int bits = (datalen*7)-bytes;
    if (bits>0)
      bytes++;
    *out=calloc(bytes,sizeof(unsigned char));

    ascii7(data,*out);

    return bytes;
}


unsigned char* BCH_encode(BCHObject* self, unsigned char* data, unsigned int datalen, unsigned char* ecc, unsigned int ecclen) {
        
    if (ecc) {
        if (ecclen != self->bch->ecc_bytes) {
            printf("ecc length must be %d bytes",
                self->bch->ecc_bytes);
            return NULL;
        }
        memcpy(self->ecc, ecc, self->bch->ecc_bytes);
    } else {
        memset(self->ecc, 0, self->bch->ecc_bytes);
    }
    
    bch_encode(self->bch, (uint8_t *) data, (unsigned int) datalen,
            self->ecc);
        
    return self->ecc;

}
     


int BCH_decode(BCHObject *self, unsigned char* data, unsigned int datalen, unsigned char* ecc) {

         
    int ecclen=self->bch->t;
    unsigned int* errloc=calloc(ecclen,sizeof(unsigned int));

/*
    for (int i=0; i<datalen; i++) {
        printf("dta %x\n",data[i]);
    }
    for (int i=0; i<self->bch->t; i++) {
        printf("ecc %x\n",ecc[i]);
    }
*/

    int nerr = bch_decode(self->bch, data, datalen, ecc, NULL, NULL, errloc);


   
    for (int i = 0; i < nerr; i++) {
		unsigned int bitnum = errloc[i];
		if (bitnum >= datalen*8 + ecclen*8) {
                        free(errloc);
			return -1;
		}
		if (bitnum < datalen*8) {
			data[bitnum/8] ^= 1 << (bitnum & 7);
		} else {
			ecc[bitnum/8 - datalen] ^= 1 << (bitnum & 7);
		}
    }

    free(errloc);
    return nerr;

}



BCHObject BCH;

int init(void) {
 
   int t=5, m = -1;
   unsigned int prim_poly = 137;
   int swap_bits = false;
 
   if (m == -1) {
        unsigned int tmp = prim_poly;
        m = 0;
        while (tmp >>= 1) {
            m++;
        }
   }

   // INIT
   if (!BCH_alloc(&BCH, m, t, prim_poly, swap_bits))  {
      printf("Alloc failed\n");
      return -1;
   }

   return 0;
}

uint8_t* inplacedecode(uint8_t* compresspacket, unsigned int datalen) {


   unsigned char* packet=calloc(datalen,sizeof(char));

   for (int i=0; i<datalen; i++) {
        packet[i]=128*compresspacket[i*2]+compresspacket[i*2+1];   
    //printf("Packetin[%d]=%x\n",i,packet[i]);
   }

   unsigned char* ecc=packet+(datalen-BCH.bch->t); 

   int nerr = BCH_decode(&BCH, packet, datalen-BCH.bch->t, ecc);

   if (nerr<0) {
     return NULL;
   }
   return packet;
}

uint8_t* inplaceencode(uint8_t* compresspacket, unsigned int datalen, unsigned int ecclen) 
{
   unsigned char* packet=calloc(datalen+ecclen,sizeof(char));

   for (int i=0; i<datalen; i++) {
        packet[i]=128*compresspacket[i*2]+compresspacket[i*2+1];   
   }

   unsigned char* ecc=packet+datalen; 

   uint8_t *ecc_return = BCH_encode(&BCH, packet, datalen, ecc, ecclen);

   for (int i=0; i<ecclen; i++) {
        packet[datalen+i]=ecc_return[i];
   }

   return packet;
}

void dealloc() {

   BCH_dealloc(&BCH);
}

