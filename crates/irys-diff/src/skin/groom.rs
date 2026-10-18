//! Strands on the Lee Perry-Smith head: eyebrows, stubble and a short scalp
//! cut, grown as procedural curves from points on the scan's surface.
//!
//! Roots are drawn area-uniformly over the mesh (in scene space, after
//! [`super::head::to_scene`]) and kept by region:
//!
//! - **scalp**: above a hairline that is a function of the azimuth about
//!   the head (forehead, temples, over the ears, the nape), ears excluded;
//! - **eyebrows**: the darkest texels of the albedo map in the brow band
//!   (the scan's brows are painted; the strands go where the paint is);
//! - **stubble**: the jaw, chin, upper lip and neck, kept with probability
//!   rising with how much darker than skin the albedo is there (the painted
//!   stubble), plus a floor so the whole beard area carries some.
//!
//! Each strand leaves its root along a comb direction projected onto the
//! surface (scalp: away from the crown; brows: outward and up; stubble:
//! down), lifted off the surface by a few degrees, bending a little toward
//! it, with noise per segment. Scalp strands are **clumped**: every strand
//! follows the nearest of a sparse set of guide strands (its azimuth turn
//! and length), plus its own noise. Deterministic in the seed.

use irys_core::{Point3, Vec3};

use super::mesh::{Mesh, Texture};
use crate::look::hair::Strands;
use crate::prb::Rng;

/// Which region a strand belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    #[allow(missing_docs)]
    Scalp,
    #[allow(missing_docs)]
    Brow,
    #[allow(missing_docs)]
    Stubble,
}

/// The grown hair.
pub struct Groom {
    /// Every segment.
    pub strands: Strands,
    /// Per segment: the melanin scale of its region.
    pub melanin: Vec<f32>,
    /// Strands per region: scalp, brows, stubble.
    pub counts: [usize; 3],
}

/// Knobs.
#[derive(Debug, Clone, Copy)]
pub struct GroomSpec {
    /// Scalp strands per unit² (1 unit = 50 mm: 2500 mm²).
    pub scalp_density: f64,
    /// Scalp hair length range, units.
    pub scalp_len: (f64, f64),
    /// Eyebrow strands per unit² where the paint is darkest.
    pub brow_density: f64,
    /// Stubble strands per unit² where the paint is darkest.
    pub stubble_density: f64,
    /// Fibre radii: scalp, brows, stubble (units).
    pub radius: [f64; 3],
    /// Melanin scale per region (on the frame's `θ`).
    pub melanin: [f32; 3],
    /// Seed.
    pub seed: u64,
}

impl Default for GroomSpec {
    fn default() -> Self {
        Self {
            scalp_density: 9_000.0,
            scalp_len: (0.09, 0.16),
            brow_density: 16_000.0,
            stubble_density: 14_000.0,
            radius: [0.0011, 0.0013, 0.0011],
            melanin: [1.0, 1.25, 0.85],
            seed: 0x6a17,
        }
    }
}

fn luma(c: [f64; 3]) -> f64 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// `v` projected onto the plane of `n`, normalised (or `fallback`).
fn tangent(v: Vec3, n: Vec3, fallback: Vec3) -> Vec3 {
    let t = v - n * v.dot(&n);
    if t.norm() > 1e-9 {
        t.normalize()
    } else {
        let f = fallback - n * fallback.dot(&n);
        f.normalize()
    }
}

/// Piecewise-linear in `|φ|` (degrees from the face direction): the
/// scalp's hairline height above the nose tip.
fn hairline(phi_deg: f64) -> f64 {
    const K: [(f64, f64); 7] = [
        (0.0, 2.3),
        (30.0, 2.2),
        (55.0, 1.75),
        (80.0, 1.15),
        (110.0, 1.0),
        (140.0, 0.1),
        (180.0, -0.9),
    ];
    let a = phi_deg.abs().min(180.0);
    for w in K.windows(2) {
        if a <= w[1].0 {
            let t = (a - w[0].0) / (w[1].0 - w[0].0);
            return w[0].1 + t * (w[1].1 - w[0].1);
        }
    }
    K[K.len() - 1].1
}

/// Grow the hair on `mesh` (scene space) with its `albedo`.
pub fn groom(mesh: &Mesh, albedo: &Texture, spec: &GroomSpec) -> Groom {
    let c = Point3::new(super::head::HEAD[0], super::head::HEAD[1], super::head::HEAD[2]);
    let tip = *mesh
        .positions
        .iter()
        .min_by(|p, q| p.y.total_cmp(&q.y))
        .expect("a mesh");
    // Triangle areas, for area-uniform roots.
    let tris: Vec<[usize; 3]> = mesh
        .indices
        .chunks_exact(3)
        .map(|t| [t[0] as usize, t[1] as usize, t[2] as usize])
        .collect();
    let mut cdf = Vec::with_capacity(tris.len());
    let mut total = 0.0;
    for t in &tris {
        let [a, b, cc] = t.map(|i| mesh.positions[i]);
        total += 0.5 * (b - a).cross(&(cc - a)).norm();
        cdf.push(total);
    }
    let mut rng = Rng::new(spec.seed, 0);
    let sample = |rng: &mut Rng| {
        let u = rng.uniform() * total;
        let k = cdf.partition_point(|&x| x < u).min(tris.len() - 1);
        let (mut b1, mut b2) = (rng.uniform(), rng.uniform());
        if b1 + b2 > 1.0 {
            b1 = 1.0 - b1;
            b2 = 1.0 - b2;
        }
        let b0 = 1.0 - b1 - b2;
        let [i, j, l] = tris[k];
        let p = Point3::origin()
            + mesh.positions[i].coords() * b0
            + mesh.positions[j].coords() * b1
            + mesh.positions[l].coords() * b2;
        let n = (mesh.normals[i] * b0 + mesh.normals[j] * b1 + mesh.normals[l] * b2).normalize();
        let uv = [
            mesh.uvs[i][0] * b0 + mesh.uvs[j][0] * b1 + mesh.uvs[l][0] * b2,
            mesh.uvs[i][1] * b0 + mesh.uvs[j][1] * b1 + mesh.uvs[l][1] * b2,
        ];
        (p, n, uv)
    };
    // The skin's luma on the face, for "darker than skin".
    let skin = 0.34;
    let mut roots: Vec<(Region, Point3, Vec3)> = Vec::new();
    let max_density = spec.scalp_density.max(spec.brow_density).max(spec.stubble_density);
    let draws = (total * max_density) as usize;
    for _ in 0..draws {
        let (p, n, uv) = sample(&mut rng);
        let u = rng.uniform();
        let q = p - c;
        let phi = q.x.atan2(-q.y).to_degrees();
        let dz = p.z - tip.z;
        let col = albedo.sample(uv[0], uv[1]);
        // Darkness on a blurred map (≈ 12 texels): the paint's density, not
        // its single dark texels (moles, pores).
        let blur = [(0.0, 0.0), (0.012, 0.0), (-0.012, 0.0), (0.0, 0.012), (0.0, -0.012)]
            .iter()
            .map(|(du, dv)| luma(albedo.sample(uv[0] + du, uv[1] + dv)))
            .sum::<f64>()
            / 5.0;
        let dark = ((skin - blur) / skin).max(0.0);
        // Ears: off the skull's oval (|x| far out at the ears' height).
        let ear = q.x.abs() > 1.72 && dz > -0.6 && dz < 1.3 && q.y.abs() < 1.0;
        let accept = |d: f64| u < d / max_density;
        // The hairline is soft: density ramps up over 0.3 units above it.
        let wiggle = 0.06 * (0.07 * phi).sin() + 0.03 * (0.19 * phi + 1.3).sin();
        let ramp = ((dz - hairline(phi) - wiggle) / 0.35).clamp(0.0, 1.0);
        if ramp > 0.0 && dz < 3.2 && !ear && n.dot(&q.normalize()) > 0.2 {
            if accept(spec.scalp_density * ramp * ramp * (3.0 - 2.0 * ramp)) {
                roots.push((Region::Scalp, p, n));
            }
        } else if phi.abs() < 60.0 && dz > 0.45 && dz < 1.2 && q.y < -1.0 {
            // Brows: the paint, where it is much darker than skin.
            let w = ((dark - 0.3) / 0.3).clamp(0.0, 1.0);
            if accept(spec.brow_density * w) {
                roots.push((Region::Brow, p, n));
            }
        } else if phi.abs() < 105.0 && dz < -0.05 && dz > -2.3 && !ear {
            // Lips: red over green; no stubble.
            let lip = col[0] > 1.6 * col[1] && dz > -0.75 && phi.abs() < 25.0;
            let w = (0.1 + (dark - 0.08) / 0.4).clamp(0.0, 1.0).powi(2);
            if !lip && accept(spec.stubble_density * w) {
                roots.push((Region::Stubble, p, n));
            }
        }
    }
    // Guides for the scalp's clumps: a sparse subset, on a hash grid.
    let cell = 0.08;
    let key = |p: Point3| ((p.x / cell).floor() as i64, (p.y / cell).floor() as i64, (p.z / cell).floor() as i64);
    let mut guides: std::collections::HashMap<(i64, i64, i64), (Point3, f64, f64)> = Default::default();
    for (r, p, _) in &roots {
        if *r == Region::Scalp {
            guides.entry(key(*p)).or_insert_with(|| {
                let turn = 0.5 * (rng.uniform() - 0.5);
                let len = spec.scalp_len.0 + (spec.scalp_len.1 - spec.scalp_len.0) * rng.uniform();
                (*p, turn, len)
            });
        }
    }
    let nearest_guide = |p: Point3| {
        let (i, j, k) = key(p);
        let mut best: Option<(f64, f64, f64)> = None;
        for di in -1..=1 {
            for dj in -1..=1 {
                for dk in -1..=1 {
                    if let Some(&(g, turn, len)) = guides.get(&(i + di, j + dj, k + dk)) {
                        let d = (g - p).norm();
                        if best.is_none_or(|b| d < b.0) {
                            best = Some((d, turn, len));
                        }
                    }
                }
            }
        }
        best.map_or((0.0, 0.12), |b| (b.1, b.2))
    };
    let crown = c + Vec3::new(0.0, 0.5, 1.6);
    let mut strands = Strands::default();
    let mut melanin = Vec::new();
    let mut counts = [0usize; 3];
    for (region, p, n) in roots {
        let (k, segs, len, lift, comb, r) = match region {
            Region::Scalp => {
                let (turn, glen) = nearest_guide(p);
                let away = tangent(p - crown, n, Vec3::new(0.0, 1.0, -1.0));
                let side = n.cross(&away);
                let t = turn + 0.25 * (rng.uniform() - 0.5);
                let comb = (away * t.cos() + side * t.sin()).normalize();
                let len = glen * (0.85 + 0.3 * rng.uniform());
                (0, 5, len, 0.22 + 0.12 * rng.uniform(), comb, spec.radius[0])
            }
            Region::Brow => {
                let q = p - c;
                let out = Vec3::new(q.x.signum(), 0.0, 0.0);
                // Medial brow hairs rise; lateral ones run out and down.
                let rise = (0.9 - 0.9 * (q.x.abs() / 1.1)).clamp(-0.2, 0.9);
                let comb = tangent(out + Vec3::new(0.0, 0.0, rise), n, out);
                let t = 0.35 * (rng.uniform() - 0.5);
                let side = n.cross(&comb);
                let comb = (comb * t.cos() + side * t.sin()).normalize();
                (1, 4, 0.08 + 0.09 * rng.uniform(), 0.08 + 0.08 * rng.uniform(), comb, spec.radius[1])
            }
            Region::Stubble => {
                let down = tangent(Vec3::new(0.0, 0.0, -1.0), n, Vec3::new(0.0, -1.0, 0.0));
                let t = 0.8 * (rng.uniform() - 0.5);
                let side = n.cross(&down);
                let comb = (down * t.cos() + side * t.sin()).normalize();
                (2, 2, 0.015 + 0.025 * rng.uniform(), 0.25 + 0.35 * rng.uniform(), comb, spec.radius[2])
            }
        };
        let mut dir = (comb * lift.cos() + n * lift.sin()).normalize();
        let mut x = p + n * (1.5 * r);
        let mut pts = vec![x];
        let step = len / segs as f64;
        for _ in 0..segs {
            // Bend toward the surface, and wander.
            let wob = Vec3::new(rng.uniform() - 0.5, rng.uniform() - 0.5, rng.uniform() - 0.5) * 0.25;
            dir = (dir - n * (0.12 * dir.dot(&n).max(0.0)) + wob).normalize();
            if dir.dot(&n) < 0.05 {
                dir = (dir + n * (0.05 - dir.dot(&n))).normalize();
            }
            x = x + dir * step;
            pts.push(x);
        }
        strands.push(&pts, r);
        melanin.extend(std::iter::repeat_n(spec.melanin[k], segs));
        counts[k] += 1;
    }
    Groom {
        strands,
        melanin,
        counts,
    }
}
