// Copyright 2024 Adobe
// All Rights Reserved.
//
// NOTICE: Adobe permits you to use, modify, and distribute this file in
// accordance with the terms of the Adobe license agreement accompanying
// it.

use std::{fmt::Display, str::FromStr};

use ndarray::{Array1, ArrayD, Axis};
use ort::{TensorValueType, Value};

const VERSION_BITS: u16 = 4;

mod bch;

#[derive(Debug)]
pub(super) struct Bits(String);

/// Error type for the `bits` module.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Something went wrong while doing inference.
    #[error("onnx error: {0}")]
    Ort(#[from] ort::Error),

    /// A character was encounted that was not a '0' or a '1'. Strings that specify bitstrings
    /// should only use these two characters.
    #[error("allowed chars are '0' and '1'")]
    InvalidChar,

    /// A bitstring has too many bits to fit in the requested version's data payload.
    #[error(
        "input bitstring ({bits} bits) has more bits than version allows ({version_allows} bits)"
    )]
    InvalidDataLength { version_allows: usize, bits: usize },

    /// A bitstring of a certain length is required, and a different length was encountered.
    #[error("must be of length 100")]
    InvalidLength,

    /// Model input/output did not have the expected dimensions.
    #[error("invalid dimensions")]
    InvalidDim,

    /// String does not represent a known version.
    #[error("invalid version")]
    InvalidVersion,

    /// Watermark is missing or corrupted.
    ///
    /// Either the image did not have a valid watermark, or too many transmission errors/image
    /// artifacts occurred such that the watermark is no longer recoverable.
    #[error("corrupt watermark")]
    CorruptWatermark,
}

impl Bits {
    /// Constructs a `Bits`, adding in the additional error correction and schema bits.
    pub(super) fn apply_error_correction_and_schema(
        mut input: String,
        version: Version,
    ) -> Result<Self, Error> {
        let data_bits: usize = version.data_bits().into();

        if input.chars().any(|c| c != '0' && c != '1') {
            return Err(Error::InvalidChar);
        }

        if input.len() > data_bits {
            return Err(Error::InvalidDataLength {
                bits: input.len(),
                version_allows: data_bits,
            });
        }

        // pad the input
        input.push_str(&"0".repeat(data_bits - input.len() + (8 - data_bits % 8)));

        // pack the input into bytes
        let data: Vec<u8> = input
            .as_bytes()
            .chunks(8)
            .map(|chunk| u8::from_str_radix(std::str::from_utf8(chunk).unwrap(), 2).unwrap())
            .collect();

        // calculate the error correction bits
        let mut ecc_state = bch::bch_init(version.allowed_bit_flips() as u32, bch::POLYNOMIAL);
        let ecc = bch::bch_encode(&mut ecc_state, &data);

        // form a bitstring from the error correction bits
        let mut error_correction: String = ecc
            .iter()
            .map(|byte| format!("{byte:08b}"))
            .collect::<Vec<String>>()
            .join("");

        // split off unneeded padding
        input.truncate(data_bits);
        error_correction.truncate(version.ecc_bits().into());

        // form the encoded string
        Ok(Self(format!(
            "{input}{error_correction}{}",
            version.bitstring()
        )))
    }

    /// Get the data out of a `Bits` by removing the error correction bits.
    pub(super) fn get_data(self) -> String {
        let version = self.get_version();
        let Self(mut s) = self;
        s.truncate(version.data_bits().into());
        s
    }

    /// Return full 100-bit encoded string (data + ecc + version bits).
    pub(super) fn into_string(self) -> String {
        self.0
    }

    /// Get the version from the bits.
    pub(super) fn get_version(&self) -> Version {
        match &self.0[98..100] {
            "00" => Version::BchSuper,
            "01" => Version::Bch5,
            "10" => Version::Bch4,
            "11" => Version::Bch3,
            _ => unreachable!(),
        }
    }

    /// Construct a `Bits` from a bitstring.
    ///
    /// This function checks for bitflips in the bitstring using the error-correcting bits, and
    /// corrects them if there are fewer bitflips than are supported by the version. As a last
    /// resort, this function checks for bitflips in the version identifier by trying all possible
    /// versions.
    fn new(s: String) -> Result<Self, Error> {
        if s.chars().any(|c| c != '0' && c != '1') {
            return Err(Error::InvalidChar);
        }

        if s.len() != 100 {
            return Err(Error::InvalidLength);
        }

        let version: Version = Version::from_bitstring(&s[96..]).unwrap_or_default();

        if let Ok(bits) = Bits::new_with_version(&s, version) {
            Ok(bits)
        } else {
            let mut versions = vec![
                Version::Bch3,
                Version::Bch4,
                Version::Bch5,
                Version::BchSuper,
            ];
            versions.retain(|v| *v != version);
            let mut res = None;
            for version in versions {
                res = Some(Bits::new_with_version(&s, version));
                if res.as_ref().unwrap().is_ok() {
                    return res.unwrap();
                }
            }
            res.unwrap()
        }
    }

    fn new_with_version(s: &str, version: Version) -> Result<Self, Error> {
        let data_bits: usize = version.data_bits().into();
        let ecc_bits: usize = version.ecc_bits().into();

        let mut data = s[..data_bits].to_string();
        let mut ecc = s[data_bits..data_bits + ecc_bits].to_string();

        // pad
        data.push_str(&"0".repeat(data_bits - data.len() + (8 - data_bits % 8)));
        ecc.push_str(&"0".repeat(ecc_bits - ecc.len() + (8 - ecc_bits % 8)));

        // pack into bytes
        let mut data: Vec<u8> = data
            .as_bytes()
            .chunks(8)
            .map(|chunk| u8::from_str_radix(std::str::from_utf8(chunk).unwrap(), 2).unwrap())
            .collect();
        let ecc: Vec<u8> = ecc
            .as_bytes()
            .chunks(8)
            .map(|chunk| u8::from_str_radix(std::str::from_utf8(chunk).unwrap(), 2).unwrap())
            .collect();

        // validate and correct
        let mut ecc_state = bch::bch_init(version.allowed_bit_flips() as u32, bch::POLYNOMIAL);
        let bitflips = bch::bch_decode(&mut ecc_state, &mut data, &ecc);

        if bitflips > version.allowed_bit_flips() {
            return Err(Error::CorruptWatermark);
        }

        // unpack data and ecc
        let mut data: String = data
            .iter()
            .map(|byte| format!("{byte:08b}"))
            .collect::<Vec<String>>()
            .join("");
        let mut ecc: String = ecc
            .iter()
            .map(|byte| format!("{byte:08b}"))
            .collect::<Vec<String>>()
            .join("");
        data.truncate(data_bits);
        ecc.truncate(ecc_bits);

        Ok(Bits(format!("{data}{ecc}{}", version.bitstring())))
    }
}

impl From<Bits> for ort::Value<TensorValueType<f32>> {
    fn from(Bits(s): Bits) -> Self {
        let floats: Vec<f32> = s
            .chars()
            .map(|c| match c {
                '0' => 0.0,
                '1' => 1.0,
                _ => unreachable!(),
            })
            .collect();

        let array = Array1::from(floats);
        Value::from_array(array.insert_axis(Axis(0))).unwrap()
    }
}

impl TryFrom<ArrayD<f32>> for Bits {
    type Error = Error;

    fn try_from(array: ArrayD<f32>) -> Result<Self, Self::Error> {
        if array.shape() != [1, 100] {
            return Err(Error::InvalidDim);
        }
        let array = array.remove_axis(Axis(0));
        let mut s = String::new();
        for bit in array.iter() {
            let c = if *bit < 0. { '0' } else { '1' };
            s.push(c);
        }

        Bits::new(s)
    }
}

/// The error correction schema
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub enum Version {
    /// Tolerates 8 bit flips
    #[default]
    BchSuper,
    /// Tolerates 5 bit flips
    Bch5,
    /// Tolerates 4 bit flips
    Bch4,
    /// Tolerates 3 bit flips
    Bch3,
}

impl Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Version::BchSuper => "BCH_SUPER",
            Version::Bch5 => "BCH_5",
            Version::Bch4 => "BCH_4",
            Version::Bch3 => "BCH_3",
        };
        write!(f, "{s}")
    }
}

impl FromStr for Version {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let version = match s {
            "BCH_SUPER" => Version::BchSuper,
            "BCH_5" => Version::Bch5,
            "BCH_4" => Version::Bch4,
            "BCH_3" => Version::Bch3,
            _ => return Err(Error::InvalidVersion),
        };

        Ok(version)
    }
}

impl Version {
    /// Get the number of allowed bit flips for this version.
    fn allowed_bit_flips(&self) -> u8 {
        match self {
            Version::BchSuper => 8,
            Version::Bch5 => 5,
            Version::Bch4 => 4,
            Version::Bch3 => 3,
        }
    }

    /// Get the number of data bits for this version.
    pub fn data_bits(&self) -> u16 {
        match self {
            Version::BchSuper => 40,
            Version::Bch5 => 61,
            Version::Bch4 => 68,
            Version::Bch3 => 75,
        }
    }

    /// Get the bitstring which indicates this version.
    fn bitstring(&self) -> String {
        match self {
            Version::BchSuper => "0000".to_owned(),
            Version::Bch5 => "0001".to_owned(),
            Version::Bch4 => "0010".to_owned(),
            Version::Bch3 => "0011".to_owned(),
        }
    }

    /// Parse a version from a bitstring.
    fn from_bitstring(s: &str) -> Result<Self, Error> {
        Ok(match s {
            "0000" => Version::BchSuper,
            "0001" => Version::Bch5,
            "0010" => Version::Bch4,
            "0011" => Version::Bch3,
            _ => return Err(Error::InvalidVersion),
        })
    }

    /// Get the number of error correcting bits for this version.
    fn ecc_bits(&self) -> u16 {
        100 - VERSION_BITS - self.data_bits()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_version() {
        let input = "1011011110011000111111000000011111011111011100000110110110111000110010101101111010011011000010000001".to_owned();
        let bits = Bits(input);
        assert_eq!(bits.get_version(), Version::Bch5);
    }

    #[test]
    fn get_data() {
        let input = "1011011110011000111111000000011111011111011100000110110110111000110010101101111010011011000010000001".to_owned();
        let bits = Bits(input);
        assert_eq!(
            bits.get_data(),
            "1011011110011000111111000000011111011111011100000110110110111"
        );
    }

    #[test]
    fn new() {
        let input = "1011011110011000111111000000011111011111011100000110110110111000110010101101111010011011000010000001".to_owned();
        let bits = Bits::new(input).unwrap();
        assert_eq!(
            bits.get_data(),
            "1011011110011000111111000000011111011111011100000110110110111"
        );
    }

    #[test]
    fn fully_corrupted() {
        let input = "0000000000000000000000000000000000000000000100000110110110111000110010101101111010011011000010000001".to_owned();
        let err = Bits::new(input).unwrap_err();
        assert_eq!(err.to_string(), "corrupt watermark");
    }

    #[test]
    fn single_bitflip() {
        let input = "0011011110011000111111000000011111011111011100000110110110111000110010101101111010011011000010000001".to_owned();
        let bits = Bits::new(input).unwrap();
        assert_eq!(
            bits.get_data(),
            "1011011110011000111111000000011111011111011100000110110110111"
        );
    }

    #[test]
    fn single_bitflip_and_corrupted_version() {
        let input = "0011011110011000111111000000011111011111011100000110110110111000110010101101111010011011000010000011".to_owned();
        let bits = Bits::new(input).unwrap();
        assert_eq!(
            bits.get_data(),
            "1011011110011000111111000000011111011111011100000110110110111"
        );
    }

    #[test]
    fn invalid_bitstring() {
        let err = Bits::apply_error_correction_and_schema("hello".to_string(), Version::Bch5)
            .unwrap_err();
        assert!(matches!(err, Error::InvalidChar));
    }

    #[test]
    fn too_long_input() {
        let err =
            Bits::apply_error_correction_and_schema("0".repeat(200), Version::Bch5).unwrap_err();
        assert!(matches!(err, Error::InvalidDataLength { .. }));
    }

    #[test]
    fn corrupt() {
        let err = Bits::new("1".repeat(100)).unwrap_err();
        assert!(matches!(err, Error::CorruptWatermark));
    }

    #[test]
    fn invalid_dim() {
        let ar = ArrayD::<f32>::zeros(ndarray::IxDyn(&[3]));
        let res: Result<Bits, _> = ar.try_into();
        assert!(matches!(res.unwrap_err(), Error::InvalidDim));
    }
}
