/* tslint:disable */
/* eslint-disable */

/**
 * Recover a metal ball's roughness and spectral F0 from a target render.
 */
export class MaterialFit {
    free(): void;
    [Symbol.dispose](): void;
    image_rgba(): Uint8Array;
    /**
     * `truth = [roughness, F0 blue, F0 green, F0 red]`.
     */
    constructor(size: number, spp: number, target_spp: number, truth: Float64Array);
    /**
     * One Adam step. Returns the loss before it.
     */
    step(): number;
    target_rgba(): Uint8Array;
    /**
     * `[roughness, F0 blue, F0 green, F0 red]`.
     */
    theta(): Float64Array;
}

/**
 * Recover the room's paint, a three-band spectral albedo, from a target.
 */
export class PaintFit {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * `[blue, green, red]`.
     */
    bands(): Float64Array;
    image_rgba(): Uint8Array;
    /**
     * `truth = [blue, green, red]` band albedos.
     */
    constructor(size: number, spp: number, target_spp: number, truth: Float64Array);
    step(): number;
    target_rgba(): Uint8Array;
}

/**
 * The paint room, progressively: one frame per call, the running mean
 * shown. All diffuse, so it needs no material graph.
 */
export class RoomRender {
    free(): void;
    [Symbol.dispose](): void;
    frame(): number;
    image_rgba(): Uint8Array;
    constructor(size: number, spp: number);
    /**
     * Repaint the room, `[blue, green, red]`, and start over.
     */
    set_paint(b: number, g: number, r: number): void;
}

/**
 * Recover every splat's colour and opacity in a bush from a target render.
 */
export class SplatFit {
    free(): void;
    [Symbol.dispose](): void;
    image_rgba(): Uint8Array;
    constructor(size: number, spp: number, target_spp: number);
    /**
     * One Adam step over all splats' DC colours and opacities.
     */
    step(): number;
    target_rgba(): Uint8Array;
}

/**
 * Rust panics reach the browser console as messages, not `unreachable`.
 */
export function start(): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_materialfit_free: (a: number, b: number) => void;
    readonly __wbg_paintfit_free: (a: number, b: number) => void;
    readonly __wbg_roomrender_free: (a: number, b: number) => void;
    readonly __wbg_splatfit_free: (a: number, b: number) => void;
    readonly materialfit_image_rgba: (a: number) => [number, number];
    readonly materialfit_new: (a: number, b: number, c: number, d: number, e: number) => number;
    readonly materialfit_step: (a: number) => number;
    readonly materialfit_target_rgba: (a: number) => [number, number];
    readonly materialfit_theta: (a: number) => [number, number];
    readonly paintfit_bands: (a: number) => [number, number];
    readonly paintfit_image_rgba: (a: number) => [number, number];
    readonly paintfit_new: (a: number, b: number, c: number, d: number, e: number) => number;
    readonly paintfit_step: (a: number) => number;
    readonly paintfit_target_rgba: (a: number) => [number, number];
    readonly roomrender_frame: (a: number) => number;
    readonly roomrender_image_rgba: (a: number) => [number, number];
    readonly roomrender_new: (a: number, b: number) => number;
    readonly roomrender_set_paint: (a: number, b: number, c: number, d: number) => void;
    readonly splatfit_image_rgba: (a: number) => [number, number];
    readonly splatfit_new: (a: number, b: number, c: number) => number;
    readonly splatfit_step: (a: number) => number;
    readonly splatfit_target_rgba: (a: number) => [number, number];
    readonly start: () => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
