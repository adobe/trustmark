/*!
 * BCH ECC Library
 * Copyright 2024 Adobe. All rights reserved.
 * Licensed under the MIT License.
 *
 * NOTICE: Adobe permits you to use, modify, and distribute this file in
 * accordance with the terms of the Adobe license agreement accompanying it.
 */

function uint8ToBoolean(u8array) {
  const resultArray = [];

  for (let i = 0; i < u8array.length; i++) {
    let c = u8array[i];
    resultArray.push((c & 0x80) > 0);
    resultArray.push((c & 0x40) > 0);
    resultArray.push((c & 0x20) > 0);
    resultArray.push((c & 0x10) > 0);
    resultArray.push((c & 0x8) > 0);
    resultArray.push((c & 0x4) > 0);
    resultArray.push((c & 0x2) > 0);
    resultArray.push((c & 0x1) > 0);
  }
  return resultArray;
}

function load4bytes(data) {
  let w = BigInt(0);
  if (data.length > 0) w += data[0] << BigInt(24);
  if (data.length > 1) w += data[1] << BigInt(16);
  if (data.length > 2) w += data[2] << BigInt(8);
  if (data.length > 3) w += data[3] << BigInt(0);
  return w;
}

function loadwordfrombits(data, lim) {
  // for example 8 or 32
  let w = BigInt(0);
  if (lim < data.length) lim = data.length;
  let pos = lim - 1;
  for (let b = 0; b < lim; b++) {
    if (data[b]) w += BigInt(1) << BigInt(pos);
    pos--;
  }
  return w;
}

function loadallwords(data, lim) {
  const datalen = data.length;
  let mlen = Math.floor(datalen / lim); // how many whole words
  let eccbuf = [];
  let offset = 0;
  while (mlen > 0) {
    let w = loadwordfrombits(data.slice(offset, offset + lim), lim);
    eccbuf.push(w);
    offset += lim;
    mlen--;
  }
  data = data.slice(offset, data.length);
  let leftdata = data.length;
  if (leftdata > 0) {
    let w = loadwordfrombits(data, lim);
    eccbuf.push(w);
  }
  return eccbuf;
}

function BCH_Encode(self, databits) {
  let data = loadallwords(databits, 8);
  const datalen = data.length;
  const l = ceilop(self.ECCstate.m * self.ECCstate.t, 32) - 1;
  let ecc = Array(self.ECCstate.ecc_bytes).fill(0);
  const ecc_max_words = ceilop(31 * 64, 32);
  let r = Array(ecc_max_words).fill(BigInt(0));

  const tab0idx = 0;
  const tab1idx = tab0idx + 256 * (l + 1);
  const tab2idx = tab1idx + 256 * (l + 1);
  const tab3idx = tab2idx + 256 * (l + 1);

  let mlen = Math.floor(datalen / 4);
  let offset = 0;
  while (mlen > 0) {
    let w = load4bytes(data.slice(offset, offset + 4));
    w = w ^ r[0];
    const p0 = tab0idx + (l + 1) * Number((w >> BigInt(0)) & BigInt(0xff));
    const p1 = tab1idx + (l + 1) * Number((w >> BigInt(8)) & BigInt(0xff));
    const p2 = tab2idx + (l + 1) * Number((w >> BigInt(16)) & BigInt(0xff));
    const p3 = tab3idx + (l + 1) * Number((w >> BigInt(24)) & BigInt(0xff));

    for (let i = 0; i < l; i++) {
      r[i] =
        r[i + 1] ^
        self.ECCstate.cyclic_tab[p0 + i] ^
        self.ECCstate.cyclic_tab[p1 + i] ^
        self.ECCstate.cyclic_tab[p2 + i] ^
        self.ECCstate.cyclic_tab[p3 + i];
    }

    r[l] =
      self.ECCstate.cyclic_tab[p0 + l] ^
      self.ECCstate.cyclic_tab[p1 + l] ^
      self.ECCstate.cyclic_tab[p2 + l] ^
      self.ECCstate.cyclic_tab[p3 + l];
    mlen--;
    offset += 4;
  }

  data = data.slice(offset, data.length);
  let leftdata = data.length;
  ecc = r;

  let posn = 0;
  while (leftdata) {
    const tmp = data[posn];
    posn++;
    let pidx = (l + 1) * Number(((ecc[0] >> BigInt(24)) ^ tmp) & BigInt(0xff));
    for (let i = 0; i < l; i++) {
      ecc[i] =
        (((ecc[i] << BigInt(8)) & BigInt(0xffffffff)) |
          (ecc[i + 1] >> BigInt(24))) ^
        self.ECCstate.cyclic_tab[pidx];
      pidx++;
    }
    ecc[l] =
      ((ecc[l] << BigInt(8)) & BigInt(0xffffffff)) ^
      self.ECCstate.cyclic_tab[pidx];
    leftdata--;
  }

  self.ECCstate.ecc_buf = ecc;
  let eccout = [];
  for (let idx = 0; idx < r.length; idx++) {
    const e = r[idx];
    eccout.push(Number(e >> BigInt(24)) & 0xff);
    eccout.push(Number(e >> BigInt(16)) & 0xff);
    eccout.push(Number(e >> BigInt(8)) & 0xff);
    eccout.push(Number(e >> BigInt(0)) & 0xff);
  }

  eccout = eccout.slice(0, self.ECCstate.ecc_bytes);

  return eccout;
}

function int8_stringify(data) {
  return data.map((num) => String.fromCharCode(Number(num))).join("");
}

function toBinaryString(num, length) {
  // Convert the number to a binary string
  let binaryString = num.toString(2);

  // Pad with leading zeros to the desired length
  while (binaryString.length < length) {
    binaryString = "0" + binaryString;
  }

  return binaryString;
}

function genBinString(dataout, datalen) {
  let out = "";
  for (let i = 0; i < dataout.length; i++) {
    out += toBinaryString(dataout[i], 8);
  }
  out = out.slice(0, datalen);
  return out;
}

function bin_stringify(data) {
  return data.join("");
}

function deepcopy(poly) {
  const newpoly = {};
  newpoly.deg = poly.deg;
  newpoly.c = Array.from(poly.c);
  return newpoly;
}

function BCH_Decode(self, data, recvecc) {
  const calc_ecc = BCH_Encode(self, data);

  const ecclen = recvecc.length;
  const eccbuf = loadallwords(recvecc, 32);

  const eccwords = ceilop(self.ECCstate.m * self.ECCstate.t, 32);

  let sum = BigInt(0);
  for (let i = 0; i < eccwords; i++) {
    self.ECCstate.ecc_buf[i] = self.ECCstate.ecc_buf[i] ^ eccbuf[i];
    sum = sum | self.ECCstate.ecc_buf[i];
  }

  let dataout = loadallwords(data, 8);
  if (sum === 0) {
    const retdata = Object();
    retdata.data_ascii = int8_stringify(dataout);
    retdata.data_binary = bin_stringify(data);
    retdata.bitflips = 0;
    retdata.valid = true;
    return retdata;
  }

  let s = self.ECCstate.ecc_bits;
  const t = self.ECCstate.t;
  const syn = Array(2 * t).fill(0);

  let m = s & 31;
  let synbuf = self.ECCstate.ecc_buf;

  if (m) {
    const newval =
      synbuf[Math.floor(s / 32)] & ~BigInt(Math.pow(2, Number(32 - m)) - 1);
    synbuf[Math.floor(s / 32)] = newval;
  }

  let synptr = 0;
  while (s > 0 || synptr === 0) {
    let poly = synbuf[synptr];
    synptr += 1;
    s -= 32;
    while (poly) {
      const i = degBigInt(self, poly);
      for (let j = 0; j < 2 * t; j += 2) {
        syn[j] = syn[j] ^ g_pow(self, (j + 1) * (i + s));
      }
      poly = poly ^ BigInt(Math.pow(2, i));
    }
  }

  for (let i = 0; i < t; i++) {
    syn[2 * i + 1] = g_sqrt(self, syn[i]);
  }

  const n = self.ECCstate.n;
  let pp = -1;
  let pd = 1;

  let pelp = Object();
  pelp.deg = 0;
  pelp.c = Array(2 * t).fill(0);
  pelp.c[0] = 1;

  let elp = Object();
  elp.deg = 0;
  elp.c = Array(2 * t).fill(0);
  elp.c[0] = 1;

  let d = syn[0];
  let elp_copy = Object();
  for (let i = 0; i < t; i++) {
    if (elp.deg > t) {
      break;
    }
    if (d) {
      const k = 2 * i - pp;
      elp_copy = deepcopy(elp);
      const tmp = g_log(self, d) + n - g_log(self, pd);
      for (let j = 0; j < pelp.deg + 1; j++) {
        if (pelp.c[j]) {
          const l = g_log(self, pelp.c[j]);
          elp.c[j + k] = elp.c[j + k] ^ g_pow(self, tmp + l);
        }
      }

      const tmpDeg = pelp.deg + k;
      if (tmpDeg > elp.deg) {
        elp.deg = tmpDeg;
        pelp = deepcopy(elp_copy);
        pd = d;
        pp = 2 * i;
      }
    }
    if (i < t - 1) {
      d = syn[2 * i + 2];
      for (let j = 1; j < elp.deg + 1; j++) {
        d = d ^ g_mul(self, elp.c[j], syn[2 * i + 2 - j]);
      }
    }
  }
  self.ECCstate.elp = elp;

  const nroots = getroots(self, dataout.length, self.ECCstate.elp);
  dataout = loadallwords(data, 8);
  const recveccout = loadallwords(recvecc, 8);
  const datalen = dataout.length;
  const nbits = datalen * 8 + self.ECCstate.ecc_bits;

  const retdata = Object();
  retdata.bitflips = nroots;
  retdata.valid = false;

  if (nroots !== self.ECCstate.elp.deg || nroots < 0 || nroots > t) {
    retdata.bitflips = -1;
    return retdata;
  }

  for (let i = 0; i < nroots; i++) {
    if (self.ECCstate.errloc[i] >= nbits) {
      return retdata; // fail
    }
    self.ECCstate.errloc[i] = nbits - 1 - self.ECCstate.errloc[i];
    self.ECCstate.errloc[i] =
      (self.ECCstate.errloc[i] & ~7) | (7 - (self.ECCstate.errloc[i] & 7));
  }

  for (let bidx = 0; bidx < self.ECCstate.errloc.length; bidx++) {
    const bitflip = Number(self.ECCstate.errloc[bidx]);

    const byte = Math.floor(bitflip / 8);
    const bit = Math.pow(2, bitflip & 7);
    if (bitflip < (dataout.length + recveccout.length) * 8) {
      if (byte < dataout.length) {
        dataout[byte] = dataout[byte] ^ BigInt(bit);
      } else {
        recveccout[byte - dataout.length] =
          recveccout[byte - dataout.length] ^ BigInt(bit);
      }
    }
  }

  // Byte alignment adds zero bits that were never transmitted.
  const padding = dataout.length * 8 - data.length;
  if (
    padding &&
    (dataout[dataout.length - 1] &
      ((BigInt(1) << BigInt(padding)) - BigInt(1))) !==
      BigInt(0)
  ) {
    retdata.bitflips = -1;
    return retdata;
  }

  const ds = genBinString(dataout, data.length);
  const correctedBits = Array.from(ds, (bit) => bit === "1");
  const reencodedEcc = BCH_Encode(self, correctedBits);
  if (
    genBinString(reencodedEcc, self.ECCstate.ecc_bits) !==
    genBinString(recveccout, self.ECCstate.ecc_bits)
  ) {
    retdata.bitflips = -1;
    return retdata;
  }

  retdata.data_ascii = int8_stringify(dataout);
  retdata.data_binary = ds;
  retdata.valid = true;

  return retdata;
}

function getroots(self, k, poly) {
  let roots = Array();

  if (poly.deg > 2) {
    k = k * 8 + self.ECCstate.ecc_bits;

    let rep = Array(self.ECCstate.t * 2).fill(0);
    let d = poly.deg;
    let l = self.ECCstate.n - g_log(self, poly.c[poly.deg]);
    for (let i = 0; i < d; i++) {
      if (poly.c[i]) {
        rep[i] = mod(self, g_log(self, poly.c[i]) + l);
      } else {
        rep[i] = -1;
      }
    }

    rep[poly.deg] = 0;
    let syn0 = g_div(self, poly.c[0], poly.c[poly.deg]);
    for (let i = self.ECCstate.n - k + 1; i < self.ECCstate.n + 1; i++) {
      let syn = syn0;
      for (let j = 1; j < poly.deg + 1; j++) {
        let m = rep[j];
        if (m >= 0) {
          syn = syn ^ g_pow(self, m + j * i);
        }
      }
      if (syn == 0) {
        roots.push(self.ECCstate.n - i);
        if (roots.length == poly.deg) {
          break;
        }
      }
    }
    if (roots.length < poly.deg) {
      self.ECCstate.errloc = [];
      return -1;
    }
  }

  if (poly.deg == 1) {
    if (poly.c[0]) {
      roots.push(
        mod(
          self,
          self.ECCstate.n -
            self.ECCstate.logarithms[poly.c[0]] +
            self.ECCstate.logarithms[poly.c[1]],
        ),
      );
    }
  }

  if (poly.deg == 2) {
    if (poly.c[0] && poly.c[1]) {
      let l0 = self.ECCstate.logarithms[poly.c[0]];
      let l1 = self.ECCstate.logarithms[poly.c[1]];
      let l2 = self.ECCstate.logarithms[poly.c[2]];

      let u = g_pow(self, l0 + l2 + 2 * (self.ECCstate.n - l1));
      let r = 0;
      let v = u;
      while (v) {
        let i = deg(self, v);
        r = r ^ self.ECCstate.elp_pre[i];
        v = v ^ Math.pow(2, i);
      }
      if ((g_sqrt(self, r) ^ r) == u) {
        roots.push(
          modn(
            self,
            2 * self.ECCstate.n - l1 - self.ECCstate.logarithms[r] + l2,
          ),
        );
        roots.push(
          modn(
            self,
            2 * self.ECCstate.n - l1 - self.ECCstate.logarithms[r ^ 1] + l2,
          ),
        );
      }
    }
  }

  self.ECCstate.errloc = roots;
  return roots.length;
}

function BCH(t, poly) {
  const retBCH = Object();
  retBCH.ECCstate = Object();

  let tmp = poly;
  let m = 0;
  while (tmp >> 1) {
    tmp = tmp >> 1;
    m += 1;
  }

  retBCH.ECCstate.m = m;
  retBCH.ECCstate.t = t;
  retBCH.ECCstate.poly = Object();

  retBCH.ECCstate.n = Math.pow(2, m) - 1;
  let words = ceilop(m * t, 32);
  retBCH.ECCstate.ecc_bytes = ceilop(m * t, 8);
  retBCH.ECCstate.cyclic_tab = new Array(words * 1024).fill(BigInt(0));
  retBCH.ECCstate.syn = new Array(2 * t).fill(0);
  retBCH.ECCstate.elp = new Array(t + 1).fill(0);
  retBCH.ECCstate.errloc = new Array(t).fill(0);

  let x = 1;
  let k = Math.pow(2, deg(retBCH, poly));
  if (k !== Math.pow(2, retBCH.ECCstate.m)) {
    return -1;
  }

  retBCH.ECCstate.exponents = new Array(1 + retBCH.ECCstate.n).fill(0);
  retBCH.ECCstate.logarithms = new Array(1 + retBCH.ECCstate.n).fill(0);
  retBCH.ECCstate.elp_pre = new Array(1 + retBCH.ECCstate.m).fill(0);

  for (let i = 0; i < retBCH.ECCstate.n; i++) {
    retBCH.ECCstate.exponents[i] = x;
    retBCH.ECCstate.logarithms[x] = i;
    if (i && x === 1) {
      return -1;
    }
    x *= 2;
    if (x & k) {
      x = x ^ poly;
    }
  }

  retBCH.ECCstate.logarithms[0] = 0;
  retBCH.ECCstate.exponents[retBCH.ECCstate.n] = 1;

  let n = 0;
  const g = Object();
  g.deg = 0;
  g.c = new Array(m * t + 1).fill(0);
  let roots = new Array(retBCH.ECCstate.n + 1).fill(0);
  let genpoly = new Array(ceilop(m * t + 1, 32)).fill(0);

  for (let i = 0; i < t; i++) {
    let r = 2 * i + 1;
    for (let j = 0; j < m; j++) {
      roots[r] = 1;
      r = mod(retBCH, 2 * r);
    }
  }

  g.deg = 0;
  g.c[0] = 1;
  for (let i = 0; i < retBCH.ECCstate.n; i++) {
    if (roots[i]) {
      let r = retBCH.ECCstate.exponents[i];
      g.c[g.deg + 1] = 1;
      for (let j = g.deg; j > 0; j--) {
        g.c[j] = g_mul(retBCH, g.c[j], r) ^ g.c[j - 1];
      }
      g.c[0] = g_mul(retBCH, g.c[0], r);
      g.deg += 1;
    }
  }

  n = g.deg + 1;
  let i = 0;

  while (n > 0) {
    let nbits;
    if (n > 32) {
      nbits = 32;
    } else {
      nbits = n;
    }

    let word = BigInt(0);
    for (let j = 0; j < nbits; j++) {
      if (g.c[n - 1 - j]) {
        word = word | BigInt(Math.pow(2, 31 - j));
      }
    }
    genpoly[i] = word;
    i += 1;
    n -= nbits;
  }

  retBCH.ECCstate.ecc_bits = g.deg;

  build_cyclic(retBCH, genpoly);

  let sum = 0;
  let aexp = 0;
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      sum = sum ^ g_pow(retBCH, i * Math.pow(2, j));
    }
    if (sum) {
      aexp = retBCH.ECCstate.exponents[i];
      break;
    }
  }

  x = 0;
  let precomp = new Array(31).fill(0);
  let remaining = m;

  while (x <= retBCH.ECCstate.n && remaining) {
    let y = g_sqrt(retBCH, x) ^ x;
    for (let i = 0; i < 2; i++) {
      let r = g_log(retBCH, y);
      if (y && r < m && !precomp[r]) {
        retBCH.ECCstate.elp_pre[r] = x;
        precomp[r] = 1;
        remaining -= 1;
        break;
      }
      y = y ^ aexp;
    }
    x += 1;
  }

  return retBCH;
}

// GALOIS OPERATORS

function g_inv(self, a) {
  return self.ECCstate.exponents[self.ECCstate.n - self.ECCstate.logarithms[a]];
}

function g_sqrt(self, a) {
  if (a) {
    return self.ECCstate.exponents[mod(self, 2 * self.ECCstate.logarithms[a])];
  } else {
    return 0;
  }
}

function mod(self, v) {
  if (v < self.ECCstate.n) {
    return v;
  } else {
    return v - self.ECCstate.n;
  }
}

function ceilop(a, b) {
  return Math.floor((a + b - 1) / b);
}

function g_mul(self, a, b) {
  if (a > 0 && b > 0) {
    let res = mod(
      self,
      self.ECCstate.logarithms[a] + self.ECCstate.logarithms[b],
    );
    return self.ECCstate.exponents[res];
  } else {
    return 0;
  }
}

function g_div(self, a, b) {
  if (a) {
    return self.ECCstate.exponents[
      mod(
        self,
        self.ECCstate.logarithms[a] +
          self.ECCstate.n -
          self.ECCstate.logarithms[b],
      )
    ];
  } else {
    return 0;
  }
}

function modn(self, v) {
  let n = self.ECCstate.n;
  while (v >= n) {
    v -= n;
    v = (v & n) + (v >> self.ECCstate.m);
  }
  return v;
}

function g_log(self, x) {
  return self.ECCstate.logarithms[x];
}

function a_ilog(self, x) {
  return mod(self, self.ECCstate.n - self.ECCstate.logarithms[x]);
}

function g_pow(self, i) {
  return self.ECCstate.exponents[modn(self, i)];
}

function deg(self, x) {
  let count = 0;
  while (x >> 1) {
    x = x >> 1;
    count += 1;
  }
  return count;
}

function degBigInt(self, x) {
  let count = 0;
  while (x >> BigInt(1)) {
    x = x >> BigInt(1);
    count += 1;
  }
  return count;
}

function build_cyclic(self, g) {
  let l = ceilop(self.ECCstate.m * self.ECCstate.t, 32);
  let plen = ceilop(self.ECCstate.ecc_bits + 1, 32);
  let ecclen = ceilop(self.ECCstate.ecc_bits, 32);

  self.ECCstate.cyclic_tab = new Array(4 * 256 * l).fill(BigInt(0));

  for (let i = 0; i < 256; i++) {
    for (let b = 0; b < 4; b++) {
      let offset = (b * 256 + i) * l;
      let data = BigInt(BigInt(i) << BigInt(8 * b));
      while (data) {
        let d = degBigInt(self, data);
        data = data ^ (g[0] >> BigInt(31 - d));

        for (let j = 0; j < ecclen; j++) {
          let hi = 0;
          let lo = 0;
          if (d < 31) {
            hi = BigInt(g[j] << BigInt(d + 1)) & BigInt(0xffffffff);
          } else {
            hi = BigInt(0);
          }
          if (j + 1 < plen) {
            lo = g[j + 1] >> BigInt(31 - d);
          } else {
            lo = BigInt(0);
          }
          let hilo = hi | lo;
          if (self.ECCstate.cyclic_tab[j + offset] == 0) {
            self.ECCstate.cyclic_tab[j + offset] = BigInt(0);
          }
          self.ECCstate.cyclic_tab[j + offset] =
            self.ECCstate.cyclic_tab[j + offset] ^ hilo;
        }
      }
    }
  }
}
