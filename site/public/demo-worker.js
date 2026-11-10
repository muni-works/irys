// One irys demo per worker: the wasm runs here so a gradient step (up to a
// second of interpreter time) never blocks the page.
import init, { SplatFit, PaintFit, RoomRender } from './pkg/irys_wasm.js';

let demo = null, kind = null, running = false, left = 0, it = 0, size = 0;
const ready = init();

function post(extra = {}) {
  const msg = { kind, it, size, img: demo.image_rgba(), ...extra };
  if (kind === 'paint') msg.params = Array.from(demo.bands());
  postMessage(msg, [msg.img.buffer]);
}

function loop() {
  if (!running || left <= 0) { running = false; postMessage({ kind, done: true, it }); return; }
  const t = performance.now();
  let loss = null;
  if (kind === 'render') it = demo.frame(); else { loss = demo.step(); it += 1; }
  left -= 1;
  post({ loss, ms: performance.now() - t });
  setTimeout(loop, 0);
}

onmessage = async (e) => {
  await ready;
  const m = e.data;
  if (m.type === 'init') {
    running = false; kind = m.kind; it = 0; size = m.size;
    if (kind === 'splat') demo = new SplatFit(m.size, m.spp, m.targetSpp);
    if (kind === 'paint') demo = new PaintFit(m.size, m.spp, m.targetSpp, new Float64Array(m.truth));
    if (kind === 'render') demo = new RoomRender(m.size, m.spp);
    if (kind !== 'render') {
      const tgt = demo.target_rgba();
      postMessage({ kind, target: tgt, size }, [tgt.buffer]);
    }
    postMessage({ kind, readyDemo: true });
  } else if (m.type === 'material') {
    demo.set_paint(...m.values); it = 0;
  } else if (m.type === 'run') {
    left = m.steps; if (!running) { running = true; loop(); }
  } else if (m.type === 'stop') {
    running = false;
  }
};
