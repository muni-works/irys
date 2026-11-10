# irys landing page

A static, framework-free page built with Vite, in the same shape as the
vcad.io, phyz.dev and tang.toys sites: one `index.html`, IBM Plex, images in
`public/`.

```sh
cd site
npm install
npm run dev      # http://localhost:4174
npm run build    # writes dist/
npm run preview  # serves dist/ on :4174
```

The live demos are `crates/irys-wasm` compiled with wasm-pack into
`public/pkg` (checked in, as phyz.dev does), run in a Web Worker
(`public/demo-worker.js`). Rebuild them off the Mac:

```sh
cd crates/irys-wasm
wasm-pack build --release --target web --out-dir ../../site/public/pkg
```

The metal material's graph runs in tang's interpreter, single-threaded in
wasm since tang `7da2ab65`, which is the rev irys pins.

Every number on the page comes from the repository README, `docs/design.md`
or the paper's `data/measured.tsv`. Images are JPEG copies of PNGs in `docs/`,
plus the paper cover (rose robot in the Barcelona Pavilion; scene by eMirage,
CC-BY; classroom by Christophe Seux, CC0).

Not done:

- crates.io: irys-core, -prims, -light, -gpu and irys-render 0.1.0 are
  published (2026-11-10). irys-diff and the research crates are not; they
  wait on tang-compile reaching crates.io.
- Domain: not chosen. Links on the page are absolute (muni.works, vcad.io,
  tang.toys, github.com/muni-works/irys), so the page works under any host.
- Deploy: none configured. `dist/` is plain static files.
- muni.works: the home page has no irys tile yet. A one-line blurb that
  matches the page: "Spectral path tracing with bit-exact replay."
- The paper link points at https://muni.works/research/irys, which must be
  live before launch.
