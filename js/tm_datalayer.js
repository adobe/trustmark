/*!
 * TrustMark Data Layer module
 * Copyright 2024 Adobe. All rights reserved.
 * Licensed under the MIT License.
 * 
 * NOTICE: Adobe permits you to use, modify, and distribute this file in
 * accordance with the terms of the Adobe license agreement accompanying it.
 */ 

// Initialize ECC engines for all supported versions
let eccengine = [];
for (let version = 0; version < 4; version++) {
  eccengine.push(DataLayer_GetECCEngine(version));
}

/**
 * Decodes the watermark data using the given ECC engine and schema.
 * Attempts fallback decoding with alternate schemas if the primary attempt fails.
 *
 * @param {Array<boolean>} watermarkbool - The boolean array representing the watermark.
 * @param {Array<Object>} eccengine - Array of ECC engines for decoding.
 * @returns {Object} Decoded watermark data with schema and soft binding info.
 */
function DataLayer_Decode(watermarkbool, eccengine, variant) {
  let version = DataLayer_GetVersion(watermarkbool);
  let databits = DataLayer_GetSchemaDataBits(version);

  let data = watermarkbool.slice(0, databits);
  let ecc = watermarkbool.slice(databits, 96);

  let dataobj = BCH_Decode(eccengine[version], data, ecc);
  dataobj.schema = DataLayer_GetSchemaName(version);

  if (!dataobj.valid) {
    // Attempt decoding with alternate schemas
    for (let alt = 0; alt < 3; alt++) {
      if (alt === version) continue;

      databits = DataLayer_GetSchemaDataBits(alt);
      data = watermarkbool.slice(0, databits);
      ecc = watermarkbool.slice(databits, 96);

      dataobj = BCH_Decode(eccengine[alt], data, ecc);
      dataobj.schema = DataLayer_GetSchemaName(alt);
      if (dataobj.valid) break;
    }
  }

  // Add soft binding information
  dataobj.softBindingInfo = formatSoftBindingData(dataobj.data_binary, version, variant);
  return dataobj;
}

/**
 * Interprets the watermark data in the context of C2PA.
 *
 * @param {Object} dataobj - The decoded watermark data object.
 * @param {number} version - The version of the schema.
 * @returns {Promise<Object>} Promise resolving with the updated data object.
 */
function interpret_C2PA(dataobj, version) {
  return new Promise((resolve, reject) => {
    if (true) { // Placeholder for schema-specific logic
      fetchSoftBindingInfo(dataobj.data_binary)
        .then(softBindingInfo => {
          if (softBindingInfo) {
            dataobj.softBindingInfo = softBindingInfo;
          } else {
            console.warn("No soft binding info found.");
          }
          resolve(dataobj);
        })
        .catch(error => {
          console.error("Error fetching soft binding info:", error);
          reject(error);
        });
    } else {
      resolve(dataobj);
    }
  });
}

/**
 * Extracts the schema version from the last two bits of the watermark boolean array.
 *
 * @param {Array<boolean>} watermarkbool - The boolean array representing the watermark.
 * @returns {number} The schema version as an integer.
 */
function DataLayer_GetVersion(watermarkbool) {
  watermarkbool = watermarkbool.slice(-2);
  return watermarkbool[0] * 2 + watermarkbool[1];
}

/**
 * Retrieves the ECC engine for the given schema version.
 *
 * @param {number} version - The schema version.
 * @returns {Object} The corresponding ECC engine.
 */
function DataLayer_GetECCEngine(version) {
  switch (version) {
    case 0:
      return BCH(8, 137);
    case 1:
      return BCH(5, 137);
    case 2:
      return BCH(4, 137);
    case 3:
      return BCH(3, 137);
    default:
      return -1;
  }
}

/**
 * Retrieves the number of data bits for the given schema version.
 *
 * @param {number} version - The schema version.
 * @returns {number} The number of data bits.
 */
function DataLayer_GetSchemaDataBits(version) {
  switch (version) {
    case 0:
      return 40;
    case 1:
      return 61;
    case 2:
      return 68;
    case 3:
      return 75;
    default:
      console.error("Invalid schema version");
      return 0;
  }
}

/**
 * Retrieves the name of the schema for the given version.
 *
 * @param {number} version - The schema version.
 * @returns {string} The schema name.
 */
function DataLayer_GetSchemaName(version) {
  switch (version) {
    case 0:
      return "BCH_SUPER";
    case 1:
      return "BCH_5";
    case 2:
      return "BCH_4";
    case 3:
      return "BCH_3";
    default:
      return "Invalid";
  }
}

/**
 * Formats the encoded watermark data into a structured JSON object.
 *
 * @param {Array<boolean>} encodedData - The binary data representing the watermark.
 * @param {number} version - The schema version.
 * @returns {Object|null} Formatted JSON object or null in case of errors.
 */
function formatSoftBindingData(encodedData, version, variant) {
  try {
    const binaryString = Array.isArray(encodedData)
      ? encodedData.join('')
      : String(encodedData);

    return {
      "c2pa.soft-binding": {
        "alg": `com.adobe.trustmark.${variant}`,
        "blocks": [
          {
            "scope": {},
            "value": `${version}*${binaryString}`
          }
        ]
      }
    };
  } catch (error) {
    console.error("Error formatting soft binding data:", error);
    return null;
  }
}

