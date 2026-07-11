// GGUF header parser/writer for the Editor section.
// TypeScript port of the standalone editor's gguf-parser.js.

const GGUF_MAGIC = 0x46554747;

export const GGUFValueType = {
  UINT8: 0,
  INT8: 1,
  UINT16: 2,
  INT16: 3,
  UINT32: 4,
  INT32: 5,
  FLOAT32: 6,
  BOOL: 7,
  STRING: 8,
  ARRAY: 9,
  UINT64: 10,
  INT64: 11,
  FLOAT64: 12,
} as const;

export const GGUFValueTypeName: Record<number, string> = {
  0: "UINT8",
  1: "INT8",
  2: "UINT16",
  3: "INT16",
  4: "UINT32",
  5: "INT32",
  6: "FLOAT32",
  7: "BOOL",
  8: "STRING",
  9: "ARRAY",
  10: "UINT64",
  11: "INT64",
  12: "FLOAT64",
};

const GGMLQuantizationType: Record<number, string> = {
  0: "F32",
  1: "F16",
  2: "Q4_0",
  3: "Q4_1",
  6: "Q5_0",
  7: "Q5_1",
  8: "Q8_0",
  9: "Q8_1",
  10: "Q2_K",
  11: "Q3_K",
  12: "Q4_K",
  13: "Q5_K",
  14: "Q6_K",
  15: "Q8_K",
  16: "IQ2_XXS",
  17: "IQ2_XS",
  18: "IQ3_XXS",
  19: "IQ1_S",
  20: "IQ4_NL",
  21: "IQ3_S",
  22: "IQ2_S",
  23: "IQ4_XS",
  24: "I8",
  25: "I16",
  26: "I32",
  27: "I64",
  28: "F64",
  29: "IQ1_M",
  30: "BF16",
  34: "TQ1_0",
  35: "TQ2_0",
  39: "MXFP4",
  40: "NVFP4",
  41: "Q1_0",
};

// [blockSize, typeSize] per ggml_type id, extracted from the bundled ggml
// build (ggml_blck_size / ggml_type_size). Needed to compute exact tensor
// byte sizes when the tensor data section is rebuilt.
const GGML_TYPE_TRAITS: Record<number, [number, number]> = {
  0: [1, 4], // F32
  1: [1, 2], // F16
  2: [32, 18], // Q4_0
  3: [32, 20], // Q4_1
  6: [32, 22], // Q5_0
  7: [32, 24], // Q5_1
  8: [32, 34], // Q8_0
  9: [32, 36], // Q8_1
  10: [256, 84], // Q2_K
  11: [256, 110], // Q3_K
  12: [256, 144], // Q4_K
  13: [256, 176], // Q5_K
  14: [256, 210], // Q6_K
  15: [256, 292], // Q8_K
  16: [256, 66], // IQ2_XXS
  17: [256, 74], // IQ2_XS
  18: [256, 98], // IQ3_XXS
  19: [256, 50], // IQ1_S
  20: [32, 18], // IQ4_NL
  21: [256, 110], // IQ3_S
  22: [256, 82], // IQ2_S
  23: [256, 136], // IQ4_XS
  24: [1, 1], // I8
  25: [1, 2], // I16
  26: [1, 4], // I32
  27: [1, 8], // I64
  28: [1, 8], // F64
  29: [256, 56], // IQ1_M
  30: [1, 2], // BF16
  34: [256, 54], // TQ1_0
  35: [256, 66], // TQ2_0
  39: [32, 17], // MXFP4
  40: [64, 36], // NVFP4
  41: [128, 18], // Q1_0
};

// dtypes a user can pick for a new zero-filled tensor
// export const ZERO_TENSOR_DTYPES = [0, 1, 30, 8, 2, 3, 6, 7, 12, 13, 14, 10, 11, 20, 23];
export const ZERO_TENSOR_DTYPES = [0, 1, 30, 8, 2, 3, 6, 7, 15, 14, 13, 12, 11, 10, 19, 29, 16, 17, 22, 18, 21, 23, 20, 39, 40];

// ─── Types ───────────────────────────────────────────────────────────────────

export interface GGUFArrayValue {
  _isArray: true;
  elemType: number;
  items: GGUFScalar[];
}

export type GGUFScalar = number | bigint | boolean | Uint8Array;
export type GGUFValue = GGUFScalar | GGUFArrayValue;

export interface MetadataEntry {
  type: number;
  value: GGUFValue;
}

export interface TensorInfo {
  name: string;
  shape: number[];
  dtype: number;
  offset: bigint;
}

export interface ParsedGGUF {
  version: number;
  metadata: Record<string, MetadataEntry>;
  tensorInfos: TensorInfo[];
  tensorDataOffset: number;
  alignment: number;
}

export interface NewMetaRow {
  key: string;
  value: string;
  // Scalar GGUF type id, or 100 + elemType for array rows.
  type: number;
}

export function typeTraits(dtype: number): [number, number] | null {
  return GGML_TYPE_TRAITS[dtype] ?? null;
}

// exact byte size of a tensor's data, mirroring ggml_row_size() * nrows
export function tensorByteSize(dtype: number, shape: number[]): number {
  const traits = GGML_TYPE_TRAITS[dtype];
  if (!traits) throw new Error(`Unknown tensor type id ${dtype}`);
  const [blockSize, typeSize] = traits;
  const ne0 = shape.length > 0 ? shape[0] : 1;
  if (ne0 % blockSize !== 0) {
    throw new Error(
      `Row size ${ne0} is not divisible by block size ${blockSize} of ${quantizationName(dtype)}`
    );
  }
  let rows = 1;
  for (let d = 1; d < shape.length; d += 1) rows *= shape[d];
  return (ne0 / blockSize) * typeSize * rows;
}

// ─── Reader ──────────────────────────────────────────────────────────────────

class GGUFReader {
  private buffer: ArrayBuffer;
  private view: DataView;
  private offset = 0;
  private le = true;
  private decoder = new TextDecoder("utf-8");

  constructor(buffer: ArrayBuffer) {
    this.buffer = buffer;
    this.view = new DataView(buffer);
  }

  private readUint8(): number {
    const value = this.view.getUint8(this.offset);
    this.offset += 1;
    return value;
  }

  private readInt8(): number {
    const value = this.view.getInt8(this.offset);
    this.offset += 1;
    return value;
  }

  private readUint16(): number {
    const value = this.view.getUint16(this.offset, this.le);
    this.offset += 2;
    return value;
  }

  private readInt16(): number {
    const value = this.view.getInt16(this.offset, this.le);
    this.offset += 2;
    return value;
  }

  private readUint32(): number {
    const value = this.view.getUint32(this.offset, this.le);
    this.offset += 4;
    return value;
  }

  private readInt32(): number {
    const value = this.view.getInt32(this.offset, this.le);
    this.offset += 4;
    return value;
  }

  private readFloat32(): number {
    const value = this.view.getFloat32(this.offset, this.le);
    this.offset += 4;
    return value;
  }

  private readFloat64(): number {
    const value = this.view.getFloat64(this.offset, this.le);
    this.offset += 8;
    return value;
  }

  private readBool(): boolean {
    return this.readUint8() !== 0;
  }

  private readUint64(): bigint {
    const lo = this.view.getUint32(this.offset, this.le);
    const hi = this.view.getUint32(this.offset + 4, this.le);
    this.offset += 8;
    return BigInt(hi) * 0x100000000n + BigInt(lo);
  }

  private readInt64(): bigint {
    const lo = this.view.getUint32(this.offset, this.le);
    const hi = this.view.getInt32(this.offset + 4, this.le);
    this.offset += 8;
    return BigInt(hi) * 0x100000000n + BigInt(lo);
  }

  private readString(): string {
    const len = Number(this.readUint64());
    if (this.offset + len > this.buffer.byteLength) {
      throw new RangeError("String extends past end of buffer");
    }
    const bytes = new Uint8Array(this.buffer, this.offset, len);
    this.offset += len;
    return this.decoder.decode(bytes);
  }

  private readRawString(): Uint8Array {
    const len = Number(this.readUint64());
    if (this.offset + len > this.buffer.byteLength) {
      throw new RangeError("String extends past end of buffer");
    }
    const bytes = new Uint8Array(this.buffer, this.offset, len).slice();
    this.offset += len;
    return bytes;
  }

  private readValue(type: number): GGUFValue {
    switch (type) {
      case GGUFValueType.UINT8:
        return this.readUint8();
      case GGUFValueType.INT8:
        return this.readInt8();
      case GGUFValueType.UINT16:
        return this.readUint16();
      case GGUFValueType.INT16:
        return this.readInt16();
      case GGUFValueType.UINT32:
        return this.readUint32();
      case GGUFValueType.INT32:
        return this.readInt32();
      case GGUFValueType.FLOAT32:
        return this.readFloat32();
      case GGUFValueType.BOOL:
        return this.readBool();
      case GGUFValueType.STRING:
        return this.readRawString();
      case GGUFValueType.ARRAY: {
        const elemType = this.readUint32();
        const count = Number(this.readUint64());
        const items: GGUFScalar[] = [];
        for (let i = 0; i < count; i += 1) {
          items.push(this.readValue(elemType) as GGUFScalar);
        }
        return { _isArray: true, elemType, items };
      }
      case GGUFValueType.UINT64:
        return this.readUint64();
      case GGUFValueType.INT64:
        return this.readInt64();
      case GGUFValueType.FLOAT64:
        return this.readFloat64();
      default:
        throw new Error(`Unknown GGUF value type: ${type}`);
    }
  }

  parse(): ParsedGGUF {
    const magic = this.readUint32();
    if (magic !== GGUF_MAGIC) throw new Error("Invalid GGUF file (bad magic bytes)");

    const version = this.readUint32();
    if (version < 1 || version > 3) throw new Error(`Unsupported GGUF version: ${version}`);

    const tensorCount = this.readUint64();
    const metadataCount = this.readUint64();
    const metadata: Record<string, MetadataEntry> = {};

    for (let i = 0; i < Number(metadataCount); i += 1) {
      const key = this.readString();
      const type = this.readUint32();
      const value = this.readValue(type);
      metadata[key] = { type, value };
    }

    const tensorInfos: TensorInfo[] = [];
    for (let i = 0; i < Number(tensorCount); i += 1) {
      const name = this.readString();
      const nDims = this.readUint32();
      const shape: number[] = [];
      for (let d = 0; d < nDims; d += 1) shape.push(Number(this.readUint64()));
      const dtype = this.readUint32();
      const offset = this.readUint64();
      tensorInfos.push({ name, shape, dtype, offset });
    }

    const metaAlign = Number(metadata["general.alignment"]?.value ?? 32);
    const alignment = Number.isFinite(metaAlign) && metaAlign > 0 ? metaAlign : 32;
    const tensorDataOffset = Math.ceil(this.offset / alignment) * alignment;

    return { version, metadata, tensorInfos, tensorDataOffset, alignment };
  }
}

// ─── Writer ──────────────────────────────────────────────────────────────────

class GGUFWriter {
  private chunks: Uint8Array[] = [];
  private encoder = new TextEncoder();

  private push(array: Uint8Array | ArrayBuffer) {
    this.chunks.push(array instanceof Uint8Array ? array : new Uint8Array(array));
  }

  writeUint8(value: number) {
    const bytes = new Uint8Array(1);
    new DataView(bytes.buffer).setUint8(0, value);
    this.push(bytes);
  }

  writeInt8(value: number) {
    const bytes = new Uint8Array(1);
    new DataView(bytes.buffer).setInt8(0, value);
    this.push(bytes);
  }

  writeUint16(value: number) {
    const bytes = new Uint8Array(2);
    new DataView(bytes.buffer).setUint16(0, value, true);
    this.push(bytes);
  }

  writeInt16(value: number) {
    const bytes = new Uint8Array(2);
    new DataView(bytes.buffer).setInt16(0, value, true);
    this.push(bytes);
  }

  writeUint32(value: number) {
    const bytes = new Uint8Array(4);
    new DataView(bytes.buffer).setUint32(0, value, true);
    this.push(bytes);
  }

  writeInt32(value: number) {
    const bytes = new Uint8Array(4);
    new DataView(bytes.buffer).setInt32(0, value, true);
    this.push(bytes);
  }

  writeFloat32(value: number) {
    const bytes = new Uint8Array(4);
    new DataView(bytes.buffer).setFloat32(0, value, true);
    this.push(bytes);
  }

  writeFloat64(value: number) {
    const bytes = new Uint8Array(8);
    new DataView(bytes.buffer).setFloat64(0, value, true);
    this.push(bytes);
  }

  writeBool(value: boolean) {
    this.writeUint8(value ? 1 : 0);
  }

  writeUint64(value: bigint | number) {
    const bytes = new Uint8Array(8);
    const view = new DataView(bytes.buffer);
    const big = BigInt(value);
    view.setUint32(0, Number(big & 0xffffffffn), true);
    view.setUint32(4, Number(big >> 32n), true);
    this.push(bytes);
  }

  writeInt64(value: bigint | number) {
    const bytes = new Uint8Array(8);
    const view = new DataView(bytes.buffer);
    const big = BigInt(value);
    view.setUint32(0, Number(big & 0xffffffffn), true);
    view.setInt32(4, Number(big >> 32n), true);
    this.push(bytes);
  }

  writeString(value: string | Uint8Array) {
    const bytes = value instanceof Uint8Array ? value : this.encoder.encode(value);
    this.writeUint64(BigInt(bytes.length));
    this.push(bytes);
  }

  writeValue(type: number, value: GGUFValue) {
    switch (type) {
      case GGUFValueType.UINT8:
        this.writeUint8(value as number);
        break;
      case GGUFValueType.INT8:
        this.writeInt8(value as number);
        break;
      case GGUFValueType.UINT16:
        this.writeUint16(value as number);
        break;
      case GGUFValueType.INT16:
        this.writeInt16(value as number);
        break;
      case GGUFValueType.UINT32:
        this.writeUint32(value as number);
        break;
      case GGUFValueType.INT32:
        this.writeInt32(value as number);
        break;
      case GGUFValueType.FLOAT32:
        this.writeFloat32(value as number);
        break;
      case GGUFValueType.BOOL:
        this.writeBool(value as boolean);
        break;
      case GGUFValueType.STRING:
        this.writeString(value as string | Uint8Array);
        break;
      case GGUFValueType.ARRAY: {
        const arr = value as GGUFArrayValue;
        this.writeUint32(arr.elemType);
        this.writeUint64(BigInt(arr.items.length));
        arr.items.forEach((item) => this.writeValue(arr.elemType, item));
        break;
      }
      case GGUFValueType.UINT64:
        this.writeUint64(value as bigint);
        break;
      case GGUFValueType.INT64:
        this.writeInt64(value as bigint);
        break;
      case GGUFValueType.FLOAT64:
        this.writeFloat64(value as number);
        break;
      default:
        throw new Error(`Unknown type: ${type}`);
    }
  }

  build(): Uint8Array {
    const total = this.chunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const out = new Uint8Array(total);
    let position = 0;
    this.chunks.forEach((chunk) => {
      out.set(chunk, position);
      position += chunk.length;
    });
    return out;
  }
}

// ─── Edited value parsing ────────────────────────────────────────────────────

function parseArrayEditedValue(elemType: number, editedStr: string): GGUFArrayValue {
  const stripped = editedStr.trim().replace(/^\[/, "").replace(/\]$/, "");
  if (!stripped) return { _isArray: true, elemType, items: [] };

  const parts = stripped
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);
  const items: GGUFScalar[] = parts.map((part) => {
    switch (elemType) {
      case GGUFValueType.FLOAT32:
      case GGUFValueType.FLOAT64:
        return parseFloat(part);
      case GGUFValueType.UINT64:
      case GGUFValueType.INT64:
        return BigInt(part);
      case GGUFValueType.BOOL:
        return part.toLowerCase() === "true";
      case GGUFValueType.STRING:
        return new TextEncoder().encode(part);
      default:
        return Number(part);
    }
  });

  return { _isArray: true, elemType, items };
}

function parseEditedValue(
  type: number,
  originalValue: GGUFValue | null,
  editedStr: string
): GGUFValue {
  try {
    if (type >= 100 && type < 120) {
      return parseArrayEditedValue(type - 100, editedStr);
    }

    switch (type) {
      case GGUFValueType.STRING:
        return new TextEncoder().encode(editedStr);
      case GGUFValueType.BOOL:
        return editedStr.trim().toLowerCase() === "true";
      case GGUFValueType.FLOAT32:
      case GGUFValueType.FLOAT64:
        return parseFloat(editedStr);
      case GGUFValueType.UINT64:
      case GGUFValueType.INT64:
        return BigInt(editedStr.trim());
      case GGUFValueType.ARRAY:
        return parseArrayEditedValue((originalValue as GGUFArrayValue).elemType, editedStr);
      default:
        return Number(editedStr);
    }
  } catch {
    return originalValue as GGUFValue;
  }
}

// ─── Header builders ─────────────────────────────────────────────────────────

export function parseGGUF(buffer: ArrayBuffer): ParsedGGUF {
  return new GGUFReader(buffer).parse();
}

function buildUpdatedMetadata(
  metadata: Record<string, MetadataEntry>,
  editedMetadata: Record<string, string>,
  deletedMetaKeys: Set<string>,
  newMetaRows: NewMetaRow[]
): Record<string, MetadataEntry> {
  const updatedMetadata: Record<string, MetadataEntry> = {};

  Object.entries(metadata).forEach(([key, entry]) => {
    if (deletedMetaKeys.has(key)) return;
    if (key in editedMetadata) {
      updatedMetadata[key] = {
        type: entry.type,
        value: parseEditedValue(entry.type, entry.value, editedMetadata[key]),
      };
      return;
    }
    updatedMetadata[key] = entry;
  });

  newMetaRows.forEach((row) => {
    const key = row.key.trim();
    if (!key || key in updatedMetadata) return;
    const actualType = row.type >= 100 ? GGUFValueType.ARRAY : row.type;
    updatedMetadata[key] = {
      type: actualType,
      value: parseEditedValue(row.type, null, row.value),
    };
  });

  return updatedMetadata;
}

export function resolveAlignment(parsedData: ParsedGGUF): number {
  const alignmentValue =
    parsedData.alignment ?? parsedData.metadata["general.alignment"]?.value ?? 32;
  return Number.isFinite(Number(alignmentValue)) && Number(alignmentValue) > 0
    ? Number(alignmentValue)
    : 32;
}

export interface FinalTensor {
  name: string;
  shape: number[];
  dtype: number;
  offset: number | bigint;
}

// builds a complete GGUF header for an explicit final tensor list
// ({ name, shape, dtype, offset }); offsets are relative to the start of the
// tensor data section.
export function buildGGUFHeaderWithTensors(
  parsedData: ParsedGGUF,
  editedMetadata: Record<string, string>,
  deletedMetaKeys: Set<string>,
  newMetaRows: NewMetaRow[],
  finalTensors: FinalTensor[]
): Uint8Array {
  const { version, metadata } = parsedData;
  const updatedMetadata = buildUpdatedMetadata(
    metadata,
    editedMetadata,
    deletedMetaKeys,
    newMetaRows
  );
  const alignment = resolveAlignment(parsedData);

  const writer = new GGUFWriter();
  writer.writeUint32(GGUF_MAGIC);
  writer.writeUint32(version);
  writer.writeUint64(BigInt(finalTensors.length));
  writer.writeUint64(BigInt(Object.keys(updatedMetadata).length));

  Object.entries(updatedMetadata).forEach(([key, { type, value }]) => {
    writer.writeString(key);
    writer.writeUint32(type);
    writer.writeValue(type, value);
  });

  finalTensors.forEach((tensor) => {
    writer.writeString(tensor.name);
    writer.writeUint32(tensor.shape.length);
    tensor.shape.forEach((dim) => writer.writeUint64(BigInt(dim)));
    writer.writeUint32(tensor.dtype);
    writer.writeUint64(BigInt(tensor.offset));
  });

  const header = writer.build();
  const paddedLen = Math.ceil(header.length / alignment) * alignment;
  const padded = new Uint8Array(paddedLen);
  padded.set(header);
  return padded;
}

export function buildGGUFHeader(
  parsedData: ParsedGGUF,
  editedMetadata: Record<string, string>,
  editedTensorNames: string[],
  deletedTensors: Set<number>,
  deletedMetaKeys: Set<string> = new Set(),
  newMetaRows: NewMetaRow[] = []
): Uint8Array {
  const filteredTensors = parsedData.tensorInfos
    .map((tensor, index) => ({ ...tensor, name: editedTensorNames[index] ?? tensor.name }))
    .filter((_, index) => !deletedTensors.has(index));

  return buildGGUFHeaderWithTensors(
    parsedData,
    editedMetadata,
    deletedMetaKeys,
    newMetaRows,
    filteredTensors
  );
}

// ─── Formatting ──────────────────────────────────────────────────────────────

const rawStringDecoder = new TextDecoder("utf-8", { fatal: false });

export function formatValue(type: number, value: GGUFValue, maxArrayElements = 25): string {
  if (type === GGUFValueType.ARRAY) {
    const arr = value as GGUFArrayValue;
    const shown = arr.items
      .slice(0, maxArrayElements)
      .map((item) => formatValue(arr.elemType, item));
    const more =
      arr.items.length > maxArrayElements ? `, … (+${arr.items.length - maxArrayElements})` : "";
    return `[${shown.join(", ")}${more}]`;
  }
  if (value instanceof Uint8Array) return rawStringDecoder.decode(value);
  if (typeof value === "bigint") return value.toString();
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number" && !Number.isInteger(value)) return value.toPrecision(7);
  return String(value);
}

export function quantizationName(dtype: number): string {
  return GGMLQuantizationType[dtype] ?? `Unknown(${dtype})`;
}
