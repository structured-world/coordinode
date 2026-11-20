// Read and write the gzip-compressed tar archives Cargo packages crates as.
//
// ustar headers with the GNU long-name (`L`) and pax (`x`) path extensions,
// which is what Cargo writes for paths beyond 100 bytes.
import { gunzipSync, gzipSync } from "node:zlib";

export interface TarMember {
  name: string;
  /** `0` regular file, `5` directory, `2` symlink, ... (the ustar type flag). */
  type: string;
  data: Buffer;
}

const BLOCK = 512;

function field(header: Buffer, offset: number, length: number): string {
  const raw = header.subarray(offset, offset + length);
  const end = raw.indexOf(0);
  return raw.subarray(0, end === -1 ? length : end).toString("utf8");
}

function octal(header: Buffer, offset: number, length: number): number {
  const text = field(header, offset, length).trim();
  return text === "" ? 0 : Number.parseInt(text, 8);
}

/** The pax `path` record of an extended header, if it has one. */
function paxPath(data: Buffer): string | undefined {
  let offset = 0;
  while (offset < data.length) {
    const space = data.indexOf(0x20, offset);
    if (space === -1) break;
    const length = Number.parseInt(data.subarray(offset, space).toString("utf8"), 10);
    if (!Number.isFinite(length) || length <= 0) break;
    const record = data.subarray(space + 1, offset + length - 1).toString("utf8");
    const equals = record.indexOf("=");
    if (record.slice(0, equals) === "path") return record.slice(equals + 1);
    offset += length;
  }
  return undefined;
}

/** Every member of a `.tar.gz` archive, in archive order. */
export function readTarGz(archive: Buffer): TarMember[] {
  const tar = gunzipSync(archive);
  const members: TarMember[] = [];
  let offset = 0;
  let longName: string | undefined;
  while (offset + BLOCK <= tar.length) {
    const header = tar.subarray(offset, offset + BLOCK);
    if (header.every((byte) => byte === 0)) break;
    const size = octal(header, 124, 12);
    const type = field(header, 156, 1) || "0";
    const data = Buffer.from(tar.subarray(offset + BLOCK, offset + BLOCK + size));
    offset += BLOCK + Math.ceil(size / BLOCK) * BLOCK;
    if (type === "L") {
      longName = data.toString("utf8").replace(/\0+$/, "");
      continue;
    }
    if (type === "x") {
      longName = paxPath(data) ?? longName;
      continue;
    }
    if (type === "g") continue;
    const prefix = field(header, 345, 155);
    const short = field(header, 0, 100);
    const name = longName ?? (prefix ? `${prefix}/${short}` : short);
    longName = undefined;
    members.push({ name, type: type === "\0" ? "0" : type, data });
  }
  return members;
}

/** A `.tar.gz` of regular files, for fixtures. Names stay within ustar's 100 bytes. */
export function writeTarGz(files: Record<string, Buffer | string>): Buffer {
  const blocks: Buffer[] = [];
  for (const [name, value] of Object.entries(files)) {
    const data = Buffer.isBuffer(value) ? value : Buffer.from(value);
    if (Buffer.byteLength(name) > 100) throw new Error(`fixture name too long: ${name}`);
    const header = Buffer.alloc(BLOCK);
    header.write(name, 0, "utf8");
    header.write("0000644\0", 100);
    header.write("0000000\0", 108);
    header.write("0000000\0", 116);
    header.write(data.length.toString(8).padStart(11, "0") + "\0", 124);
    header.write("00000000000\0", 136);
    header.write("        ", 148);
    header.write("0", 156);
    header.write("ustar\0", 257);
    header.write("00", 263);
    let sum = 0;
    for (const byte of header) sum += byte;
    header.write(sum.toString(8).padStart(6, "0") + "\0 ", 148);
    blocks.push(header, data, Buffer.alloc((BLOCK - (data.length % BLOCK)) % BLOCK));
  }
  blocks.push(Buffer.alloc(BLOCK * 2));
  return gzipSync(Buffer.concat(blocks));
}
