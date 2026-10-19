// irys-ffi harness: make an IOSurface, render into it through the C ABI
// (no copy), read its pixels on the CPU, and compare with the readback path.
// Headless: no window, no layer — the surface is just memory both sides share.
import Foundation
import IOSurface

let w = 320, h = 240
guard let ctx = irys_context_create() else {
    print("skip: \(String(cString: irys_last_error()))"); exit(0)
}
let ply = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : nil
let scene: OpaquePointer? = ply.map { p in p.withCString { irys_scene_load(ctx, nil, $0, UInt32(w), UInt32(h)) } }
    ?? irys_scene_load(ctx, nil, nil, UInt32(w), UInt32(h))
guard let scene else { print("load: \(String(cString: irys_last_error()))"); exit(1) }

let props: [IOSurfacePropertyKey: Any] = [
    .width: w, .height: h, .bytesPerElement: 4,
    .pixelFormat: 0x5247_4241,   // 'RGBA'
]
guard let surface = IOSurface(properties: props) else { print("IOSurface failed"); exit(1) }

if ply == nil {
    // the demo scene's own camera; a splat scene needs one
} else {
    let rot: [Double] = [1, 0, 0, 0, 0, 1, 0, -1, 0]   // looking along +y, z up
    let pos: [Double] = [0, -2, 0.9]
    _ = irys_scene_set_camera(scene, 220, 220, 159.5, 119.5, rot, pos)
}

let spp: UInt32 = 8
var t0 = Date()
let ok = irys_scene_render_iosurface(scene, Unmanaged.passUnretained(surface).toOpaque(), spp)
guard ok == 0 else { print("render: \(String(cString: irys_last_error()))"); exit(1) }
let first = Date().timeIntervalSince(t0)
t0 = Date()
for _ in 0..<10 { _ = irys_scene_render_iosurface(scene, Unmanaged.passUnretained(surface).toOpaque(), spp) }
let per = Date().timeIntervalSince(t0) / 10
// and through the id
guard irys_scene_render_iosurface_id(scene, IOSurfaceGetID(surface), spp) == 0 else {
    print("render by id: \(String(cString: irys_last_error()))"); exit(1)
}

surface.lock(options: .readOnly, seed: nil)
let row = surface.bytesPerRow
let base = surface.baseAddress.assumingMemoryBound(to: UInt8.self)
var mine = [UInt8](repeating: 0, count: w * h * 4)
for y in 0..<h { for x in 0..<w * 4 { mine[y * w * 4 + x] = base[y * row + x] } }
surface.unlock(options: .readOnly, seed: nil)

var ref = [UInt8](repeating: 0, count: w * h * 4)
guard irys_scene_render_rgba8(scene, &ref, spp) == 0 else { print("readback failed"); exit(1) }
var maxd = 0, over = 0, sum = 0
for i in 0..<mine.count {
    let d = abs(Int(mine[i]) - Int(ref[i]))
    maxd = max(maxd, d); if d > 1 { over += 1 }; sum += Int(mine[i])
}
print(String(format: "IOSurface %dx%d (row %d B): first frame %.1f ms, then %.2f ms/frame at %d spp; mean %.1f; vs readback max |d| %d, %d bytes > 1",
             w, h, row, first * 1e3, per * 1e3, spp, Double(sum) / Double(mine.count), maxd, over))
irys_scene_destroy(scene)
irys_context_destroy(ctx)
exit(maxd <= 1 && sum > 0 ? 0 : 1)
