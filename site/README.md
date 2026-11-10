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

Every number on the page comes from the repository README, `docs/design.md`
or the paper's `data/measured.tsv`. Images are JPEG copies of PNGs in `docs/`,
plus the paper cover (rose robot in the Barcelona Pavilion; scene by eMirage,
CC-BY; classroom by Christophe Seux, CC0).

Not done:

- Domain: not chosen. Links on the page are absolute (muni.works, vcad.io,
  tang.toys, github.com/muni-works/irys), so the page works under any host.
- Deploy: none configured. `dist/` is plain static files.
- muni.works: the home page has no irys tile yet. A one-line blurb that
  matches the page: "Spectral path tracing with bit-exact replay."
- The paper link points at https://muni.works/research/irys, which must be
  live before launch.
