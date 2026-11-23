// tiny-stack-vm: a small, exact interpreter for make.fun's tiny stack ISA (v1).
// One run answers a whole set: for each /input/<id>/ it runs program.bin on
// input.bin and writes /output/<id>/stdout.bin and exit.txt. Node 22 standard
// library only, no dependencies, fully deterministic (no clock, no randomness).
// The step order is the reference VM's, because it decides which trap fires:
// (1) step limit, (2) pc outside the program, (3) unknown opcode,
// (4) immediate cut off by the program's end, (5) stack underflow,
// (6) stack overflow, then the instruction's own traps and effect. All
// arithmetic wraps to 32 bits; MUL keeps the low 32 bits of the exact product.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

// Opcode table: one packed byte per opcode, (immediate bytes + 1) | pops << 3
// | pushes << 5, so dispatch costs one table read a step. The +1 marks known
// opcodes: an unknown byte reads 0 and traps 5. Built from the ISA's 37
// opcodes: [byte, immediate bytes + 1, values popped, values pushed].
const INFO: [number, number, number, number][] = [
  [0, 1, 0, 0], [1, 2, 0, 1], [2, 5, 0, 1], [3, 1, 1, 0], [4, 1, 1, 2],
  [5, 1, 2, 2], [6, 1, 2, 3], [7, 1, 3, 3],
  [16, 1, 2, 1], [17, 1, 2, 1], [18, 1, 2, 1], [19, 1, 2, 1], [20, 1, 2, 1],
  [21, 1, 2, 1], [22, 1, 2, 1], [23, 1, 1, 1], [24, 1, 2, 1], [25, 1, 2, 1],
  [26, 1, 2, 1], [27, 1, 1, 1], [28, 1, 2, 1], [29, 1, 2, 1], [30, 1, 2, 1],
  [32, 1, 2, 1], [33, 1, 2, 1], [34, 1, 2, 1],
  [48, 3, 0, 0], [49, 3, 1, 0], [50, 3, 1, 0], [51, 3, 0, 0], [52, 1, 0, 0],
  [64, 1, 1, 1], [65, 1, 2, 0],
  [80, 1, 1, 0], [81, 1, 1, 0], [82, 1, 0, 1], [83, 1, 1, 0],
];
const T = new Uint8Array(256);
for (const [op, imm, pops, pushes] of INFO) T[op] = imm | (pops << 3) | (pushes << 5);

function run(p: Uint8Array, inp: Uint8Array): { out: Uint8Array; code: number } {
  const n = p.length;
  const D = new Int32Array(256); // data stack
  let sp = 0;
  const C = new Int32Array(64); // call stack
  let cs = 0;
  const M = new Int32Array(256); // memory, 256 words
  const out = new Uint8Array(65536);
  let ol = 0;
  let ip = 0;
  let pc = 0;
  let steps = 0;
  let code = 0;

  outer: for (;;) {
    if (steps === 1000000) { code = 10; break; }
    if (pc < 0 || pc >= n) { code = 9; break; }
    const op = p[pc];
    const t = T[op];
    if (!t) { code = 5; break; }
    const imm = (t & 7) - 1;
    if (pc + imm >= n) { code = 9; break; }
    steps++;
    const pops = (t >> 3) & 3;
    if (sp < pops) { code = 2; break; }
    if (sp - pops + ((t >> 5) & 3) > 256) { code = 3; break; }
    const nx = pc + 1 + imm;
    pc = nx;
    switch (op) {
      case 0: // HALT
        break outer;
      case 1: // PUSH8, sign-extended
        D[sp++] = (p[nx - 1] << 24) >> 24;
        break;
      case 2: // PUSH32, little-endian
        D[sp++] = p[nx - 4] | (p[nx - 3] << 8) | (p[nx - 2] << 16) | (p[nx - 1] << 24);
        break;
      case 3: // POP
        sp--;
        break;
      case 4: // DUP
        D[sp] = D[sp - 1];
        sp++;
        break;
      case 5: { // SWAP
        const t = D[sp - 1];
        D[sp - 1] = D[sp - 2];
        D[sp - 2] = t;
        break;
      }
      case 6: // OVER
        D[sp] = D[sp - 2];
        sp++;
        break;
      case 7: { // ROT
        const a = D[sp - 3];
        D[sp - 3] = D[sp - 2];
        D[sp - 2] = D[sp - 1];
        D[sp - 1] = a;
        break;
      }
      case 23: // NEG: -(-2147483648) stays -2147483648
        D[sp - 1] = -D[sp - 1] | 0;
        break;
      case 27: // NOT
        D[sp - 1] = ~D[sp - 1];
        break;
      case 16: case 17: case 18: case 19: case 20: case 21: case 22:
      case 24: case 25: case 26: case 28: case 29: case 30: case 32:
      case 33: case 34: {
        const b = D[sp - 1];
        const a = D[sp - 2];
        if ((op === 19 || op === 20 || op === 21 || op === 22) && b === 0) {
          code = 4; // division by zero
          break outer;
        }
        let r: number;
        switch (op) {
          case 16: r = (a + b) | 0; break;
          case 17: r = (a - b) | 0; break;
          case 18: r = Math.imul(a, b); break;
          case 19: r = a === -2147483648 && b === -1 ? a : (a / b) | 0; break;
          case 20: r = b === -1 ? 0 : (a % b) | 0; break;
          case 21: r = Math.floor((a >>> 0) / (b >>> 0)) | 0; break;
          case 22: r = ((a >>> 0) % (b >>> 0)) | 0; break;
          case 24: r = a & b; break;
          case 25: r = a | b; break;
          case 26: r = a ^ b; break;
          case 28: r = a << (b & 31); break;
          case 29: r = a >> (b & 31); break;
          case 30: r = (a >>> (b & 31)) | 0; break;
          case 32: r = a === b ? 1 : 0; break;
          case 33: r = a < b ? 1 : 0; break;
          default: r = (a >>> 0) < (b >>> 0) ? 1 : 0; // LTU
        }
        sp -= 2;
        D[sp++] = r;
        break;
      }
      case 48: case 49: case 50: { // JMP, JZ, JNZ
        const rel = ((p[nx - 2] | (p[nx - 1] << 8)) << 16) >> 16;
        let take = true;
        if (op !== 48) {
          const v = D[--sp];
          take = op === 49 ? v === 0 : v !== 0;
        }
        if (take) pc = nx + rel;
        break;
      }
      case 51: // CALL
        if (cs === 64) { code = 7; break outer; }
        C[cs++] = nx;
        pc = p[nx - 2] | (p[nx - 1] << 8);
        break;
      case 52: // RET
        if (cs === 0) { code = 8; break outer; }
        pc = C[--cs];
        break;
      case 64: { // LOAD
        const addr = D[sp - 1];
        if (addr < 0 || addr > 255) { code = 6; break outer; }
        D[sp - 1] = M[addr];
        break;
      }
      case 65: { // STORE
        const addr = D[sp - 1];
        if (addr < 0 || addr > 255) { code = 6; break outer; }
        M[addr] = D[sp - 2];
        sp -= 2;
        break;
      }
      case 80: { // PRINT, decimal ASCII, no newline
        const s = String(D[sp - 1]);
        if (ol + s.length > 65536) { code = 11; break outer; }
        sp--;
        for (let i = 0; i < s.length; i++) out[ol++] = s.charCodeAt(i);
        break;
      }
      case 81: // PUTC
        if (ol + 1 > 65536) { code = 11; break outer; }
        out[ol++] = D[--sp] & 255;
        break;
      case 82: // IN, -1 once the input is used up
        D[sp++] = ip < inp.length ? inp[ip++] : -1;
        break;
      case 83: // EXIT, low byte of the value
        code = D[--sp] & 255;
        break outer;
      default:
        code = 5; // unreachable: the table already rejected unknown opcodes
        break outer;
    }
  }
  return { out: out.subarray(0, ol), code };
}

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
for (const id of readdirSync(I).sort()) {
  const r = run(readFileSync(join(I, id, 'program.bin')), readFileSync(join(I, id, 'input.bin')));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'stdout.bin'), r.out);
  writeFileSync(join(O, id, 'exit.txt'), r.code + '\n');
}
