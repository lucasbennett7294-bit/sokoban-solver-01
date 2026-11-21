// Sokoban solver: push-optimal, room by room.
//
// The levels this bounty draws are a hallway with one to four rooms off it,
// each room holding 3 or 4 boxes and as many goals, joined to the hallway by
// a single door. A box pushed out of a door into the hallway can never be
// pushed back, so in every optimal solution each box stays in its room and
// the level's push-optimal count is the sum of its rooms' optima. This
// program therefore splits a level into rooms, solves each room on its own
// with a push-space A* search, and walks the player between rooms through
// the hallway. Every search is budgeted by work (states expanded), never by
// the clock, so two runs write the same bytes.
//
// The room search works in push space: a state is the boxes' squares plus
// the player's reachable region (named by its lowest square, which makes
// every square the player can walk to without pushing the same state). Its
// estimate is the cheapest assignment of boxes to goals over the room's
// floor (a Hungarian matching on shortest-path distances, which no push can
// beat by more than one), so the first solved state A* takes off its queue
// is push-optimal. Dead states are pruned: a box on a square no pull can
// ever bring to a goal, a box frozen in a 2 x 2 block of walls and boxes,
// and a box frozen on both axes. A greedy best-first pass and a plain
// breadth-first pass stand by if the A* pass ever runs out of its budget.
//
// Run: node main.ts (Node 22). It reads /input/<id>/level.txt for every
// instance and writes /output/<id>/solution.txt; MAKEFUN_INPUT_DIR and
// MAKEFUN_OUTPUT_DIR stand in for /input and /output outside a container.
// A solution is the player's moves in LURD, lower case a step, upper case
// a push; an empty file answers a level no search could solve.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The four steps, in a fixed order so every run ties the same way. */
const DIRS: { dx: number; dy: number; step: string; push: string }[] = [
  { dx: 0, dy: -1, step: 'u', push: 'U' },
  { dx: 0, dy: 1, step: 'd', push: 'D' },
  { dx: -1, dy: 0, step: 'l', push: 'L' },
  { dx: 1, dy: 0, step: 'r', push: 'R' },
];

/** Work budgets, in states: the whole run's search never reads the clock. */
const ASTAR_STATE_CAP = 250_000;
const GREEDY_STATE_CAP = 250_000;
const BFS_STATE_CAP = 250_000;
let workLeft = 1_500_000;

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  goalSet: Uint8Array;
  boxes: number[];
  player: number;
}

interface Room {
  squares: number[];
  boxes: number[];
  goals: number[];
  done: boolean;
}

interface Solved {
  pushes: number;
  moves: string;
}

/**
 * A level in XSB text (# wall, space floor, . goal, $ box, * box on a goal,
 * @ player, + player on a goal). Short rows and the outer frame are walls.
 */
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
      const ch = row[x]!;
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if ('.*+'.includes(ch)) goals.push(i);
      if ('$*'.includes(ch)) boxes.push(i);
      if ('@+'.includes(ch)) player = i;
    }
  });
  const goalSet = new Uint8Array(w * h);
  for (const g of goals) goalSet[g] = 1;
  return { w, h, wall, goals, goalSet, boxes: boxes.sort((a, b) => a - b), player };
}

/**
 * The squares a box can ever stand on and still reach a goal: pulled back
 * from the goals over open floor, ignoring the other boxes. A box anywhere
 * else is lost, and a push onto such a square is never part of a solution.
 */
function liveSquares(level: Level): Uint8Array {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const d of DIRS) {
      const dd = d.dy * w + d.dx;
      const y = x - dd;
      if (!wall[y] && !wall[y - dd] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/**
 * The squares the player reaches from `from` with `occupied` in the way,
 * marked in `seen` with `stamp`; returns the lowest square reached. `stack`
 * is scratch space the caller owns.
 */
function reachRegion(
  level: Level,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  const { w, wall } = level;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    for (const d of DIRS) {
      const q = p + d.dy * w + d.dx;
      if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/** Whether the box just pushed to `t` sits in a frozen 2 x 2 block. */
function frozenBlock(level: Level, occupied: Uint8Array, t: number): boolean {
  const { w, wall, goalSet } = level;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b])) continue;
    if (!(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goalSet[a]) || (occupied[b] && !goalSet[b])) return true;
    if ((occupied[c] && !goalSet[c]) || (occupied[d] && !goalSet[d])) return true;
  }
  return false;
}

/**
 * Whether some box off its goal can never move again: frozen on both axes,
 * where a side counts as blocked only by a wall or a box already frozen.
 * Cascades to a fixpoint, so a box held only by movable boxes stays free.
 */
function anyBoxFrozen(level: Level, boxes: number[], occupied: Uint8Array): boolean {
  const { w, wall, goalSet } = level;
  const frozen = new Uint8Array(wall.length);
  const blocked = (c: number): boolean => wall[c] === 1 || (occupied[c] === 1 && frozen[c] === 1);
  let changed = true;
  while (changed) {
    changed = false;
    for (const b of boxes) {
      if (goalSet[b] || frozen[b]) continue;
      const stuckH = blocked(b - 1) && blocked(b + 1);
      const stuckV = blocked(b - w) && blocked(b + w);
      if (stuckH && stuckV) {
        frozen[b] = 1;
        changed = true;
      }
    }
  }
  return boxes.some((b) => !goalSet[b] && frozen[b] === 1);
}

/** All orders of n goals, for the cheapest assignment of boxes to goals. */
function permutations(n: number): number[][] {
  const out: number[][] = [];
  const cur: number[] = [];
  const used = new Array<boolean>(n).fill(false);
  const build = (): void => {
    if (cur.length === n) {
      out.push([...cur]);
      return;
    }
    for (let i = 0; i < n; i++) {
      if (used[i]) continue;
      used[i] = true;
      cur.push(i);
      build();
      cur.pop();
      used[i] = false;
    }
  };
  build();
  return out;
}

/** The shortest walk from `from` to `to` around walls and boxes, in lower LURD. */
function walkTo(level: Level, occupied: Uint8Array, from: number, to: number): string {
  const { w, wall } = level;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const d of DIRS) {
      const q = p + d.dy * w + d.dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = d.step;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: the push square is out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

/** The square the player stands on after `moves` (case aside) from `from`. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

type SearchMode = 'astar' | 'greedy' | 'bfs';

/**
 * One room, push-optimally when mode is astar: A* over pushes with the
 * assignment estimate, dead-square and frozen-box pruning, and the player's
 * reachable region folded into each state's key. Greedy and breadth-first
 * runs of the same search stand by as fallbacks. Returns null when the
 * search runs out of its state budget or the room has no solution.
 */
function searchRoom(level: Level, mode: SearchMode, stateCap: number): Solved | null {
  const { w, wall, goals, goalSet } = level;
  const size = wall.length;
  const live = liveSquares(level);
  const solved = (bs: number[]): boolean => bs.every((b) => goalSet[b] === 1);
  if (level.boxes.some((b) => !live[b])) return null;
  if (solved(level.boxes)) return { pushes: 0, moves: '' };

  // Shortest-path distances from each goal over open floor, for the estimate.
  const dist: Int32Array[] = goals.map((g) => {
    const dd = new Int32Array(size).fill(-1);
    dd[g] = 0;
    const queue = [g];
    for (let qi = 0; qi < queue.length; qi++) {
      const p = queue[qi]!;
      for (const d of DIRS) {
        const q = p + d.dy * w + d.dx;
        if (!wall[q] && dd[q] === -1) {
          dd[q] = dd[p]! + 1;
          queue.push(q);
        }
      }
    }
    return dd;
  });
  const n = level.boxes.length;
  const perms = permutations(n);
  const estimate = (bs: number[]): number => {
    let best = Infinity;
    for (const perm of perms) {
      let c = 0;
      for (let i = 0; i < n; i++) {
        const d = dist[perm[i]!]![bs[i]!]!;
        if (d < 0) {
          c = Infinity;
          break;
        }
        c += d;
      }
      if (c < best) best = c;
    }
    return best;
  };

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  // The expansion's region keeps its own marks: the key computation below
  // reuses `seen` and would otherwise overwrite them mid-expansion.
  const regionSeen = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  let regionStamp = 0;
  const keyOf = (bs: number[], low: number): number => {
    let k = low;
    for (const b of bs) k = k * size + b;
    return k;
  };

  // The states, in the order found: boxes, the player's square after the
  // push that made them, their parent, the push that made them, and g / h.
  const stBoxes: number[][] = [level.boxes];
  const stPlayer: number[] = [level.player];
  const stParent: number[] = [-1];
  const stFrom: number[] = [-1];
  const stTo: number[] = [-1];
  const stG: number[] = [0];
  const stH: number[] = [estimate(level.boxes)];
  const stKey: number[] = [];
  const stF: number[] = [];
  const stSeq: number[] = [];
  const best = new Map<number, number>();
  for (const b of level.boxes) occupied[b] = 1;
  const firstLow = reachRegion(level, occupied, level.player, seen, ++stamp, stack);
  for (const b of level.boxes) occupied[b] = 0;
  stKey[0] = keyOf(level.boxes, firstLow);
  best.set(stKey[0]!, 0);
  const rank = (s: number): number => (mode === 'astar' ? stG[s]! + stH[s]! : mode === 'greedy' ? stH[s]! : stG[s]!);
  stF[0] = rank(0);
  stSeq[0] = 0;

  // A binary heap on (f, h, order found): the smallest f first, the most
  // promising first on a tie, the earliest found first after that.
  const heap: number[] = [0];
  let seq = 1;
  const less = (a: number, b: number): boolean =>
    stF[a]! < stF[b]! || (stF[a]! === stF[b]! && (stH[a]! < stH[b]! || (stH[a]! === stH[b]! && stSeq[a]! < stSeq[b]!)));
  const heapPush = (s: number): void => {
    let i = heap.length;
    heap.push(s);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): number => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };

  let expansions = 0;
  let found = -1;
  while (heap.length > 0) {
    const s = heapPop();
    if (best.get(stKey[s]!) !== s) continue;
    if (solved(stBoxes[s]!)) {
      found = s;
      break;
    }
    if (expansions >= stateCap || workLeft <= 0) return null;
    expansions++;
    workLeft--;
    const boxes = stBoxes[s]!;
    const g = stG[s]!;
    for (const b of boxes) occupied[b] = 1;
    const region = ++regionStamp;
    reachRegion(level, occupied, stPlayer[s]!, regionSeen, region, stack);
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi]!;
      for (const d of DIRS) {
        const dd = d.dy * w + d.dx;
        const t = b + dd;
        if (wall[t] || occupied[t] || !live[t]) continue;
        if (regionSeen[b - dd] !== region) continue;
        const next = boxes.slice();
        next[bi] = t;
        next.sort((x, y) => x - y);
        occupied[b] = 0;
        occupied[t] = 1;
        const dead = frozenBlock(level, occupied, t) || anyBoxFrozen(level, next, occupied);
        let key = -1;
        if (!dead) {
          const low = reachRegion(level, occupied, b, seen, ++stamp, stack);
          key = keyOf(next, low);
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (dead) continue;
        const prev = best.get(key);
        if (prev !== undefined && stG[prev]! <= g + 1) continue;
        const h = estimate(next);
        if (h === Infinity) continue;
        const idx = stBoxes.length;
        stBoxes.push(next);
        stPlayer.push(b);
        stParent.push(s);
        stFrom.push(b);
        stTo.push(t);
        stG.push(g + 1);
        stH.push(h);
        stKey.push(key);
        stF.push(mode === 'astar' ? g + 1 + h : mode === 'greedy' ? h : g + 1);
        stSeq.push(seq++);
        best.set(key, idx);
        heapPush(idx);
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;

  // The path of pushes, then the moves: walk to each push's square, push.
  const path: number[] = [];
  for (let s = found; s > 0; s = stParent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = level.player;
  let boxes = level.boxes;
  for (const s of path) {
    const nextBoxes = stBoxes[s]!;
    const from = stFrom[s]!;
    const to = stTo[s]!;
    const d = to - from;
    const dir = DIRS.find((x) => x.dy * w + x.dx === d)!;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(level, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += dir.push;
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves };
}

/**
 * The level's rooms: cut the hallway row's dead squares (a box there
 * reaches no goal) out of the floor; each part left that holds a box is a
 * room. Null when some part holds more boxes than goals, a goal lies in the
 * hallway, or a box starts there: then the level is no set of rooms.
 */
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const d of DIRS) {
        const q = squares[i]! + d.dy * w + d.dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [], done: false });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) if (part[g]! >= 0) rooms[part[g]!]!.goals.push(g);
  if (goals.some((g) => part[g]! < 0)) return null;
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** One room, with the fallbacks in turn; null when none of them solves it. */
function solveRoom(level: Level): Solved | null {
  return (
    searchRoom(level, 'astar', ASTAR_STATE_CAP) ??
    searchRoom(level, 'greedy', GREEDY_STATE_CAP) ??
    searchRoom(level, 'bfs', BFS_STATE_CAP)
  );
}

/**
 * A level, room by room: each room is solved on a board where the other
 * rooms' boxes stand as walls (on their goals once their room is done),
 * and the player walks on from where the last room left it. Null when a
 * room gives up, or the level is no set of rooms and the whole-level search
 * gives up too.
 */
function solveLevel(level: Level): Solved | null {
  const rooms = roomsOf(level);
  if (rooms === null) return solveRoom(level);
  let player = level.player;
  let pushes = 0;
  let moves = '';
  for (const room of rooms) {
    const wall = level.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
    }
    const sub: Level = {
      w: level.w,
      h: level.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      goalSet: level.goalSet,
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const solved = solveRoom(sub);
    if (solved === null) return null;
    pushes += solved.pushes;
    moves += solved.moves;
    player = endOf(level.w, player, solved.moves);
    room.done = true;
  }
  return { pushes, moves };
}

for (const id of readdirSync(I).sort()) {
  let moves = '';
  try {
    const solved = solveLevel(parseLevel(readFileSync(join(I, id, 'level.txt'), 'utf8')));
    if (solved !== null) moves = solved.moves;
  } catch {
    moves = '';
  }
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
