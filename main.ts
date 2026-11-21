// Sokoban solver: room-by-room push-optimal A*.
//
// A level is a one-square-wide hallway with 1-4 rooms off it (3-4 boxes each).
// A box pushed out of a room into the hallway can never come back, so every
// room is an independent problem and the level optimum is the sum of the
// rooms' optima. Each room is solved on its own with A* over pushes:
//
//   - a state is the sorted box squares plus the player's reachable region
//     (named by its lowest square, which identifies the region exactly);
//   - the heuristic is the minimum assignment of boxes to goals, where the
//     cost of one box to one goal is the fewest pushes for a lone box
//     (reverse-push BFS, ignoring other boxes): admissible, and consistent,
//     so the first solved state A* pops is push-optimal;
//   - dead squares (reverse-push reachability from the goals), frozen 2x2
//     blocks of walls and boxes, and boxes frozen by walls alone prune
//     states no push can save;
//   - a solved room must also let the player walk back out of its door, so
//     the next room can be reached (the goal test checks that the player's
//     region reaches the square outside the door);
//   - if optimal A* runs out of its work budget, a weighted/greedy pass and
//     then a plain push BFS try the same room, so a room still gets solved
//     (a few pushes over optimum beats an unsolved level);
//   - every budget is counted in expansions, never in wall-clock time, and
//     every iteration order is fixed, so two runs write the same bytes.
//
// Only Node builtins are used. No clock, no randomness, no environment reads
// besides the input/output directories.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The four moves, always in this order (deterministic tie-breaking). */
const DIRS = [
  { dx: 0, dy: -1, letter: 'u', push: 'U' },
  { dx: 0, dy: 1, letter: 'd', push: 'D' },
  { dx: -1, dy: 0, letter: 'l', push: 'L' },
  { dx: 1, dy: 0, letter: 'r', push: 'R' },
] as const;

/** Work budgets, counted in state expansions (deterministic, never time). */
const ROOM_BUDGET = 250_000; // expansions per attempt of one room
const ROOM_TOTAL_BUDGET = 700_000; // expansions across all attempts of one room
const WHOLE_BUDGET = 400_000; // expansions for the whole-level fallback
const RUN_BUDGET = 4_000_000; // expansions across the whole run (safety valve)

type Level = {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
};

type Room = {
  boxes: number[];
  goals: number[];
  exitSquare: number;
};

type Work = { used: number; cap: number };

/** A level in XSB text: its width, walls, goals, boxes and player. */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  return { w, h, wall, goals: goals.sort((a, b) => a - b), boxes: boxes.sort((a, b) => a - b), player };
}

/**
 * For each goal, the fewest pushes that move one lone box from any square
 * onto that goal (reverse-push BFS: from a square a box reaches, the square
 * one step back is reachable when the square behind that one is open floor
 * for the player to stand on). A square no goal's table reaches is dead: no
 * push sequence, with or without other boxes, can ever bring a box from it
 * onto a goal.
 */
function pushDistances(w: number, wall: Uint8Array, goals: number[]): Int32Array[] {
  const size = wall.length;
  const out: Int32Array[] = [];
  for (const g of goals) {
    const dist = new Int32Array(size).fill(-1);
    dist[g] = 0;
    const queue = [g];
    for (let qi = 0; qi < queue.length; qi++) {
      const x = queue[qi]!;
      for (let k = 0; k < 4; k++) {
        const d = DIRS[k]!.dy * w + DIRS[k]!.dx;
        const y = x - d;
        if (wall[y] || wall[y - d] || dist[y] >= 0) continue;
        dist[y] = dist[x]! + 1;
        queue.push(y);
      }
    }
    out.push(dist);
  }
  return out;
}

/** The squares from which some goal is reachable by pushes. */
function liveSquares(dists: Int32Array[]): Uint8Array {
  const size = dists[0]!.length;
  const live = new Uint8Array(size);
  for (let s = 0; s < size; s++) {
    for (const d of dists) {
      if (d[s]! >= 0) {
        live[s] = 1;
        break;
      }
    }
  }
  return live;
}

/**
 * The level's rooms: cut the hallway row's dead floor squares out of the
 * floor (a box there reaches no goal), and each connected part left that
 * holds boxes is a room. Null when the level is no set of rooms: a box
 * starts in the hallway, a goal is cut off, a part's boxes and goals differ,
 * or the player shares a part with a box.
 */
function roomsOf(level: Level): Room[] | null {
  const { w, h, wall, goals, boxes, player } = level;
  const size = w * h;
  const dists = pushDistances(w, wall, goals);
  const live = liveSquares(dists);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(size);
  for (let x = 0; x < w; x++) {
    const i = row * w + x;
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(size).fill(-1);
  const parts: { squares: number[]; boxes: number[]; goals: number[] }[] = [];
  for (let s = 0; s < size; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = parts.length;
    for (let i = 0; i < squares.length; i++) {
      for (let k = 0; k < 4; k++) {
        const q = squares[i]! + DIRS[k]!.dy * w + DIRS[k]!.dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = parts.length;
          squares.push(q);
        }
      }
    }
    parts.push({ squares, boxes: [], goals: [] });
  }
  if (part[player] >= 0) return null;
  for (const b of boxes) parts[part[b]!]!.boxes.push(b);
  for (const g of goals) {
    if (part[g] < 0) return null;
    parts[part[g]!]!.goals.push(g);
  }
  if (parts.some((p) => p.boxes.length !== p.goals.length)) return null;
  const rooms: Room[] = [];
  for (let r = 0; r < parts.length; r++) {
    const p = parts[r]!;
    if (p.boxes.length === 0) continue;
    // The square outside the door: a floor square next to the room that is
    // not part of it (the hallway square the door opens onto).
    let exitSquare = -1;
    outer: for (const s of p.squares) {
      for (let k = 0; k < 4; k++) {
        const q = s + DIRS[k]!.dy * w + DIRS[k]!.dx;
        if (!wall[q] && part[q] !== r) {
          exitSquare = q;
          break outer;
        }
      }
    }
    rooms.push({ boxes: p.boxes, goals: p.goals, exitSquare });
  }
  return rooms;
}

/** The state key: the region's lowest square, then the sorted boxes. */
function encodeKey(low: number, boxes: number[], size: number): number | string {
  if (Math.pow(size, boxes.length + 1) < Number.MAX_SAFE_INTEGER) {
    let k = low;
    for (const b of boxes) k = k * size + b;
    return k;
  }
  return `${low}:${boxes.join(',')}`;
}

function decodeKey(key: number | string, n: number, size: number): { low: number; boxes: number[] } {
  if (typeof key === 'number') {
    const boxes = new Array<number>(n);
    let k = key;
    for (let i = n - 1; i >= 0; i--) {
      boxes[i] = k % size;
      k = Math.floor(k / size);
    }
    return { low: k, boxes };
  }
  const [low, rest] = key.split(':');
  return { low: Number(low), boxes: rest!.split(',').map(Number) };
}

/** The square the player stands on after `moves` (lowercase LURD) from `from`. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

/**
 * Push-optimal (A*) / greedy / BFS search over pushes of one (sub-)level.
 * `mode` picks the priority: astar (g + h), greedy (h), bfs (g). The
 * heuristic is the minimum assignment of boxes to goals by lone-box push
 * distance (admissible and consistent), so A* is push-optimal. Returns the
 * pushes and the full move string, or null when the budget runs out.
 */
function solvePushes(
  sub: Level,
  mode: 'astar' | 'greedy' | 'bfs',
  requireExit: boolean,
  exitSquare: number,
  maxExpansions: number,
  work: Work,
  runWork: Work,
): { pushes: number; moves: string } | null {
  const { w, wall, goals, boxes: startBoxes, player: startPlayer } = sub;
  const size = wall.length;
  const n = startBoxes.length;
  const goalSet = new Uint8Array(size);
  for (const g of goals) goalSet[g] = 1;
  const dists = pushDistances(w, wall, goals);
  const live = liveSquares(dists);
  if (startBoxes.some((b) => !live[b])) return null;

  // Minimum-assignment heuristic over the boxes (sorted) to the goals.
  const popcount = new Uint8Array(1 << n);
  for (let m = 1; m < popcount.length; m++) popcount[m] = popcount[m >> 1]! + (m & 1);
  const heuristic = (boxes: number[]): number => {
    const dp = new Float64Array(1 << n).fill(Infinity);
    dp[0] = 0;
    for (let mask = 0; mask < (1 << n); mask++) {
      const base = dp[mask]!;
      if (!Number.isFinite(base)) continue;
      const i = popcount[mask]!;
      if (i >= n) continue;
      const b = boxes[i]!;
      for (let j = 0; j < n; j++) {
        if (mask & (1 << j)) continue;
        const d = dists[j]![b]!;
        if (d < 0) continue;
        const nm = mask | (1 << j);
        const v = base + d;
        if (v < dp[nm]!) dp[nm] = v;
      }
    }
    return dp[(1 << n) - 1]!;
  };

  // Flood-fill scratch: `seen` stamps the player's region, `stack` its work.
  const seen = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  const occupied = new Uint8Array(size);
  const region = (from: number): number => {
    const s = ++stamp;
    let low = from;
    let sp = 0;
    stack[sp++] = from;
    seen[from] = s;
    while (sp > 0) {
      const p = stack[--sp]!;
      if (p < low) low = p;
      for (let k = 0; k < 4; k++) {
        const q = p + DIRS[k]!.dy * w + DIRS[k]!.dx;
        if (!wall[q] && !occupied[q] && seen[q] !== s) {
          seen[q] = s;
          stack[sp++] = q;
        }
      }
    }
    return low;
  };

  // A box frozen by walls alone (no direction has both an open destination
  // and an open square for the player to stand on) can never move again.
  const wallFrozen = (t: number): boolean => {
    if (goalSet[t]) return false;
    for (let k = 0; k < 4; k++) {
      const d = DIRS[k]!.dy * w + DIRS[k]!.dx;
      if (!wall[t + d] && !wall[t - d]) return false;
    }
    return true;
  };
  // A frozen 2x2 block of walls and boxes holding a box off its goal.
  const frozenBlock = (t: number): boolean => {
    for (const a of [t, t - 1, t - w, t - w - 1]) {
      const b = a + 1;
      const c = a + w;
      const d = c + 1;
      if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
      if ((occupied[a] && !goalSet[a]) || (occupied[b] && !goalSet[b]) || (occupied[c] && !goalSet[c]) || (occupied[d] && !goalSet[d])) return true;
    }
    return false;
  };

  // The states: their key, parent, the push that made them, and g (pushes).
  const sKey: (number | string)[] = [];
  const sParent: number[] = [];
  const sFrom: number[] = [];
  const sTo: number[] = [];
  const sDir: number[] = [];
  const sG: number[] = [];
  const best = new Map<number | string, number>();
  const closed = new Set<number | string>();

  for (const b of startBoxes) occupied[b] = 1;
  const low0 = region(startPlayer);
  const key0 = encodeKey(low0, startBoxes, size);
  for (const b of startBoxes) occupied[b] = 0;
  sKey.push(key0);
  sParent.push(-1);
  sFrom.push(-1);
  sTo.push(-1);
  sDir.push(-1);
  sG.push(0);
  best.set(key0, 0);

  // Binary heap on [f, g, counter, state]: f up, then g down (deeper first),
  // then insertion order. All deterministic.
  const heap: [number, number, number, number][] = [];
  let counter = 0;
  const less = (a: [number, number, number, number], b: [number, number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      const tmp = heap[i]!;
      heap[i] = heap[p]!;
      heap[p] = tmp;
      i = p;
    }
  };
  const heapPop = (): [number, number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        const tmp = heap[i]!;
        heap[i] = heap[c]!;
        heap[c] = tmp;
        i = c;
      }
    }
    return top;
  };
  const fOf = (g: number, h: number): number => (mode === 'bfs' ? g : mode === 'greedy' ? h : g + h);
  heapPush([fOf(0, heuristic(startBoxes)), 0, counter++, 0]);

  const solved = (boxes: number[]): boolean => boxes.every((b) => goalSet[b]);

  // Shortest walk from `from` to `to` around walls and boxes, lower LURD.
  const walkTo = (from: number, to: number): string | null => {
    if (from === to) return '';
    const prev = new Int32Array(size).fill(-1);
    const how = new Array<string>(size);
    const queue = [from];
    prev[from] = from;
    for (let qi = 0; qi < queue.length; qi++) {
      const p = queue[qi]!;
      if (p === to) break;
      for (let k = 0; k < 4; k++) {
        const q = p + DIRS[k]!.dy * w + DIRS[k]!.dx;
        if (wall[q] || occupied[q] || prev[q] !== -1) continue;
        prev[q] = p;
        how[q] = DIRS[k]!.letter;
        queue.push(q);
      }
    }
    if (prev[to] === -1) return null;
    let s = '';
    for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
    return s;
  };

  let expanded = 0;
  while (heap.length > 0) {
    const [, g, , idx] = heapPop();
    const key = sKey[idx]!;
    if (closed.has(key) || best.get(key) !== idx) continue;
    if (expanded >= maxExpansions || work.used >= work.cap || runWork.used >= runWork.cap) return null;
    expanded++;
    work.used++;
    runWork.used++;
    closed.add(key);
    const { boxes } = decodeKey(key, n, size);
    const player = idx === 0 ? startPlayer : sFrom[idx]!;
    for (const b of boxes) occupied[b] = 1;
    const regionStamp = ++stamp;
    // The region again, keeping this stamp for the stand-square tests.
    let low = player;
    let sp = 0;
    stack[sp++] = player;
    seen[player] = regionStamp;
    while (sp > 0) {
      const p = stack[--sp]!;
      if (p < low) low = p;
      for (let k = 0; k < 4; k++) {
        const q = p + DIRS[k]!.dy * w + DIRS[k]!.dx;
        if (!wall[q] && !occupied[q] && seen[q] !== regionStamp) {
          seen[q] = regionStamp;
          stack[sp++] = q;
        }
      }
    }
    void low;
    if (solved(boxes) && (!requireExit || exitSquare < 0 || seen[exitSquare] === regionStamp)) {
      // Reconstruct the pushes, then the full walk between them.
      const pushes: { from: number; dir: number }[] = [];
      for (let s = idx; s > 0; s = sParent[s]!) pushes.push({ from: sFrom[s]!, dir: sDir[s]! });
      pushes.reverse();
      let moves = '';
      let p = startPlayer;
      const cur = startBoxes.slice();
      for (const b of boxes) occupied[b] = 0;
      for (const b of cur) occupied[b] = 1;
      for (const push of pushes) {
        const d = DIRS[push.dir]!.dy * w + DIRS[push.dir]!.dx;
        const stand = push.from - d;
        const walk = walkTo(p, stand);
        if (walk === null) {
          for (const b of cur) occupied[b] = 0;
          return null;
        }
        moves += walk;
        moves += DIRS[push.dir]!.push;
        occupied[push.from] = 0;
        occupied[push.from + d] = 1;
        const at = cur.indexOf(push.from);
        cur[at] = push.from + d;
        cur.sort((a, b) => a - b);
        p = push.from;
      }
      for (const b of cur) occupied[b] = 0;
      return { pushes: pushes.length, moves };
    }
    // Candidate pushes: the player must reach the stand square, and the
    // destination must be open floor, live, and unoccupied.
    const candidates: { bi: number; dir: number; dest: number }[] = [];
    for (let bi = 0; bi < n; bi++) {
      const b = boxes[bi]!;
      for (let k = 0; k < 4; k++) {
        const d = DIRS[k]!.dy * w + DIRS[k]!.dx;
        const dest = b + d;
        if (seen[b - d] !== regionStamp || wall[dest] || occupied[dest] || !live[dest]) continue;
        candidates.push({ bi, dir: k, dest });
      }
    }
    for (const cand of candidates) {
      const b = boxes[cand.bi]!;
      const dest = cand.dest;
      occupied[b] = 0;
      occupied[dest] = 1;
      if (wallFrozen(dest) || frozenBlock(dest)) {
        occupied[dest] = 0;
        occupied[b] = 1;
        continue;
      }
      const next = boxes.slice();
      next[cand.bi] = dest;
      next.sort((a, b2) => a - b2);
      const low2 = region(b);
      const key2 = encodeKey(low2, next, size);
      occupied[dest] = 0;
      occupied[b] = 1;
      const g2 = g + 1;
      const prev = best.get(key2);
      if (prev !== undefined && sG[prev]! <= g2) continue;
      const h2 = heuristic(next);
      if (!Number.isFinite(h2)) continue;
      const t = sKey.length;
      sKey.push(key2);
      sParent.push(idx);
      sFrom.push(b);
      sTo.push(dest);
      sDir.push(cand.dir);
      sG.push(g2);
      best.set(key2, t);
      heapPush([fOf(g2, h2), g2, counter++, t]);
    }
    for (const b of boxes) occupied[b] = 0;
  }
  return null;
}

/** The moves solving one level, or '' when it cannot be solved. */
function solve(text: string, runWork: Work): string {
  const level = parseLevel(text);
  const goalSet = new Set(level.goals);
  if (level.boxes.length === 0 || level.boxes.every((b) => goalSet.has(b))) return '';
  const rooms = roomsOf(level);
  if (rooms === null) {
    // No room structure: one search over the whole level.
    const work: Work = { used: 0, cap: WHOLE_BUDGET };
    const solved = solvePushes(level, 'astar', false, -1, WHOLE_BUDGET, work, runWork);
    return solved === null ? '' : solved.moves;
  }
  let player = level.player;
  let moves = '';
  for (const room of rooms) {
    // The other rooms' boxes stand as walls while this room is solved.
    const wall = level.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const b of other.boxes) wall[b] = 1;
    }
    const sub: Level = { w: level.w, h: level.h, wall, goals: room.goals, boxes: room.boxes, player };
    const work: Work = { used: 0, cap: ROOM_TOTAL_BUDGET };
    let solved: { pushes: number; moves: string } | null = null;
    const attempts: ['astar' | 'greedy' | 'bfs', boolean][] = [
      ['astar', true],
      ['astar', false],
      ['greedy', true],
      ['greedy', false],
      ['bfs', false],
    ];
    for (const [mode, exit] of attempts) {
      if (work.used >= work.cap) break;
      solved = solvePushes(sub, mode, exit, room.exitSquare, ROOM_BUDGET, work, runWork);
      if (solved !== null) break;
    }
    if (solved === null) return '';
    moves += solved.moves;
    player = endOf(level.w, player, solved.moves);
    room.boxes = room.goals.slice();
  }
  return moves;
}

const runWork: Work = { used: 0, cap: RUN_BUDGET };
for (const id of readdirSync(I).sort()) {
  let moves = '';
  try {
    moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'), runWork);
  } catch {
    moves = '';
  }
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
