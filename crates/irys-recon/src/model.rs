//! Render a program in every view, with or without its Jacobian.

use crate::dsl::Program;
use crate::geom;
use crate::jac::{self, ViewJac};
use crate::render::{self, Film, KernelCache, View};

/// The cameras, the film and the sample budgets.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Setup {
    /// Known cameras.
    pub views: Vec<View>,
    /// Model film.
    pub film: Film,
    /// Edge samples per view for the Jacobian.
    pub edge_samples: usize,
    /// Seed of the model frames (fixed: common random numbers across θ).
    pub seed: u64,
    /// Coverage supersampling per axis.
    pub coverage_sub: usize,
}

impl Setup {
    /// The v0 defaults over `views`.
    pub fn new(views: Vec<View>) -> Self {
        Self {
            views,
            film: Film { size: (256, 192), spp: 16 },
            edge_samples: 40_000,
            seed: 7,
            coverage_sub: 3,
        }
    }
}

/// One evaluation.
pub struct Eval {
    /// Grey image per view.
    pub img: Vec<Vec<f64>>,
    /// Part coverage per view.
    pub cov: Vec<Vec<f64>>,
    /// Jacobians per view, when asked for.
    pub jac: Option<Vec<ViewJac>>,
    /// Triangles.
    pub tris: usize,
}

/// Kernel caches, one per view (each view renders on its own thread).
pub struct Renderer {
    caches: Vec<KernelCache>,
}

impl Renderer {
    /// For `n` views.
    pub fn new(n: usize) -> Self {
        Self { caches: (0..n).map(|_| KernelCache::default()).collect() }
    }

    /// Render `prog` in every view of `setup`; with `jac`, its Jacobian.
    pub fn eval(&mut self, prog: &Program, setup: &Setup, jac: bool) -> Result<Eval, String> {
        let solid = prog.build()?;
        let mesh = geom::mesh(&solid);
        let np = prog.params().len();
        let vel = if jac { geom::velocities(prog, &mesh.0) } else { Vec::new() };
        let outs: Vec<(Vec<f64>, Vec<f64>, Option<ViewJac>)> = std::thread::scope(|s| {
            let hs: Vec<_> = setup
                .views
                .iter()
                .zip(self.caches.iter_mut())
                .enumerate()
                .map(|(vi, (v, k))| {
                    let (mesh, vel) = (&mesh, &vel);
                    s.spawn(move || {
                        let mut sc = render::scene(Some(mesh), v, setup.film);
                        let seed = setup.seed + 1000 * vi as u64;
                        let img = render::grey(&sc, seed, k);
                        let cov = render::coverage(&sc, setup.coverage_sub);
                        let j = jac.then(|| {
                            jac::edge_jacobian(&mut sc, &mesh.0, &mesh.1, vel, np, setup.edge_samples, seed + 17, k)
                        });
                        (img, cov, j)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut e = Eval { img: Vec::new(), cov: Vec::new(), jac: jac.then(Vec::new), tris: mesh.1.len() };
        for (i, c, j) in outs {
            e.img.push(i);
            e.cov.push(c);
            if let (Some(v), Some(j)) = (e.jac.as_mut(), j) {
                v.push(j);
            }
        }
        Ok(e)
    }
}

/// Render the target: `prog` at `spp` with its own seeds.
pub fn target(prog: &Program, setup: &Setup, spp: usize, seed: u64) -> Result<Vec<Vec<f64>>, String> {
    let solid = prog.build()?;
    let mesh = geom::mesh(&solid);
    let film = Film { size: setup.film.size, spp };
    Ok(std::thread::scope(|s| {
        let hs: Vec<_> = setup
            .views
            .iter()
            .enumerate()
            .map(|(vi, v)| {
                let mesh = &mesh;
                s.spawn(move || {
                    let sc = render::scene(Some(mesh), v, film);
                    render::grey(&sc, seed + 7919 * vi as u64, &mut KernelCache::default())
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    }))
}

/// The empty studio in every view (the background plate).
pub fn background(setup: &Setup, spp: usize) -> Vec<Vec<f64>> {
    setup
        .views
        .iter()
        .map(|v| {
            let sc = render::scene(None, v, Film { size: setup.film.size, spp });
            render::grey(&sc, 3, &mut KernelCache::default())
        })
        .collect()
}
