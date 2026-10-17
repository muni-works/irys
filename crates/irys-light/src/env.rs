//! Environments the renderer can build for itself: synthesised studio HDRIs,
//! and a lat-long Radiance `.hdr` reader.
//!
//! # Why the built-ins are generated rather than shipped
//!
//! The obvious move is to vendor a couple of Poly Haven CC0 HDRIs. These are
//! synthesised instead, for three reasons: the crate carries no binary blobs
//! and no third-party licence to track, the maps are exactly as
//! high-frequency as the sampler needs to be exercised (crisp softbox discs,
//! a hot rim), and a studio environment is a handful of soft rectangles in a
//! dim room — precisely the thing that is cheaper to describe than to store.
//! Real-world HDRIs are fully supported by [`parse_hdr`], which is the path
//! any Poly Haven download takes.
//!
//! # Why the `.hdr` reader is hand-written
//!
//! `image` would do it, but this crate's dependency rule is tang, rayon,
//! wgpu, bytemuck and pollster — nothing else. Radiance RGBE is a header, a
//! resolution line and either flat or run-length-encoded scanlines; the whole
//! decoder is under two hundred lines and builds for wasm without a thought.

use crate::math::Vec3;
use crate::pathtrace::EnvMap;

/// Resolution of the generated built-in maps. Enough to keep the softbox
/// edges clean without making CDF construction show up in a profile.
const BUILTIN_W: usize = 256;
const BUILTIN_H: usize = 128;

/// Sin-weighted mean luminance the built-ins are normalised to, so switching
/// between them changes the *look* and not the exposure.
pub const BUILTIN_MEAN: f32 = 0.30;

/// The generated studio environments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinEnv {
    /// Neutral three-light studio: broad key, cool fill, hot rim.
    Studio,
    /// Two crisp softboxes against near-black — a product-shot look with
    /// strong specular shapes.
    Softbox,
    /// Bright, even overcast dome. Low contrast, flattering to complex parts.
    Overcast,
}

impl BuiltinEnv {
    /// Lower-case spelling, as a CLI would take it.
    pub fn name(self) -> &'static str {
        match self {
            BuiltinEnv::Studio => "studio",
            BuiltinEnv::Softbox => "softbox",
            BuiltinEnv::Overcast => "overcast",
        }
    }

    /// Every built-in, for help text and tests.
    pub fn all() -> [BuiltinEnv; 3] {
        [
            BuiltinEnv::Studio,
            BuiltinEnv::Softbox,
            BuiltinEnv::Overcast,
        ]
    }

    /// Parse a lower-case built-in name.
    pub fn parse(s: &str) -> Option<Self> {
        BuiltinEnv::all().into_iter().find(|b| b.name() == s)
    }
}

// ─── generated studio maps ────────────────────────────────────────────────

/// A soft-edged disc of light centred on `dir`, of angular radius `radius`
/// radians.
fn disc(d: Vec3, dir: Vec3, radius: f64, radiance: [f32; 3]) -> [f32; 3] {
    let a = d.dot(dir.normalize()).clamp(-1.0, 1.0).acos();
    if a >= radius {
        return [0.0; 3];
    }
    let t = (1.0 - a / radius) as f32;
    let k = t * t * (3.0 - 2.0 * t);
    [radiance[0] * k, radiance[1] * k, radiance[2] * k]
}

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Radiance of a built-in environment in direction `d`, before normalisation.
fn builtin_radiance(kind: BuiltinEnv, d: Vec3) -> [f32; 3] {
    let z = d.z as f32;
    match kind {
        BuiltinEnv::Studio => {
            // Dim room: slightly cool above, warm floor bounce below.
            let base = if z >= 0.0 {
                let k = z.powf(0.7);
                [0.14 + 0.05 * k, 0.15 + 0.07 * k, 0.17 + 0.11 * k]
            } else {
                let k = (-z).powf(0.5);
                [0.14 - 0.08 * k, 0.135 - 0.08 * k, 0.13 - 0.08 * k]
            };
            let key = disc(d, Vec3::new(-0.8, -1.0, 1.1), 0.34, [26.0, 25.0, 23.5]);
            let fill = disc(d, Vec3::new(1.3, -0.6, 0.25), 0.50, [2.6, 2.9, 3.5]);
            let rim = disc(d, Vec3::new(0.35, 1.25, 0.8), 0.13, [60.0, 59.0, 57.0]);
            add(add(base, key), add(fill, rim))
        }
        BuiltinEnv::Softbox => {
            // Near-black surround so the specular shapes read hard.
            let base = if z >= 0.0 {
                [0.020, 0.021, 0.024]
            } else {
                [0.012, 0.012, 0.013]
            };
            let a = disc(d, Vec3::new(-0.7, -1.0, 0.5), 0.30, [55.0, 54.0, 52.0]);
            let b = disc(d, Vec3::new(0.9, -0.9, 0.35), 0.20, [22.0, 23.0, 26.0]);
            let c = disc(d, Vec3::new(0.1, 1.1, 0.55), 0.10, [70.0, 69.0, 68.0]);
            add(add(base, a), add(b, c))
        }
        BuiltinEnv::Overcast => {
            let base = if z >= 0.0 {
                let k = z.powf(0.6);
                [0.55 + 0.45 * k, 0.57 + 0.47 * k, 0.60 + 0.50 * k]
            } else {
                let k = (-z).powf(0.5);
                [0.30 - 0.18 * k, 0.29 - 0.17 * k, 0.28 - 0.16 * k]
            };
            // A brighter break in the cloud, so there is something for the
            // CDF to find and for gloss to catch.
            let sun = disc(d, Vec3::new(-0.4, -0.7, 1.0), 0.45, [3.2, 3.2, 3.1]);
            add(base, sun)
        }
    }
}

/// Generate a built-in map, normalised to a common mean radiance.
pub fn generate(kind: BuiltinEnv) -> EnvMap {
    let (w, h) = (BUILTIN_W, BUILTIN_H);
    let mut pixels = vec![[0.0f32; 3]; w * h];
    // Solid-angle-weighted mean, so normalisation is over the sphere and not
    // over image area (which would over-count the poles).
    let mut weighted = 0.0f64;
    let mut weight = 0.0f64;

    for j in 0..h {
        let theta = core::f64::consts::PI * (j as f64 + 0.5) / h as f64;
        let (st, ct) = theta.sin_cos();
        for i in 0..w {
            let phi = core::f64::consts::TAU * (i as f64 + 0.5) / w as f64;
            let d = Vec3::new(st * phi.cos(), st * phi.sin(), ct);
            let c = builtin_radiance(kind, d);
            pixels[j * w + i] = c;
            let lum = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
            weighted += lum as f64 * st;
            weight += st;
        }
    }

    let mean = if weight > 0.0 {
        (weighted / weight) as f32
    } else {
        0.0
    };
    if mean > 0.0 {
        let k = BUILTIN_MEAN / mean;
        for p in &mut pixels {
            p[0] *= k;
            p[1] *= k;
            p[2] *= k;
        }
    }

    EnvMap::new(w, h, pixels).expect("built-in environment dimensions are valid")
}

// ─── Radiance .hdr ────────────────────────────────────────────────────────

/// One RGBE quadruple to linear RGB.
///
/// The `e == 0` case is exact zero, not `2^-128`: Radiance writes an all-zero
/// pixel that way and the naive formula would turn it into a denormal.
#[inline]
fn rgbe_to_rgb(p: [u8; 4]) -> [f32; 3] {
    if p[3] == 0 {
        return [0.0; 3];
    }
    let f = libm_exp2(p[3] as i32 - (128 + 8));
    [
        (p[0] as f32 + 0.5) * f,
        (p[1] as f32 + 0.5) * f,
        (p[2] as f32 + 0.5) * f,
    ]
}

/// `2^n` for a small signed `n`, without `powi` on a wasm target's libm.
#[inline]
fn libm_exp2(n: i32) -> f32 {
    let n = n.clamp(-126, 127);
    f32::from_bits((((n + 127) as u32) & 0xFF) << 23)
}

/// Decode a lat-long Radiance `.hdr` (RGBE) image from its bytes.
///
/// The image is taken as equirectangular with row 0 at the zenith, which is
/// the convention every HDRI library ships. Both scanline encodings are
/// handled: flat RGBE, and the "new" adaptive run-length encoding.
pub fn parse_hdr(bytes: &[u8]) -> Result<EnvMap, String> {
    // ── header ──
    let mut i = 0usize;
    let mut line = Vec::new();
    let mut saw_magic = false;
    let resolution: (usize, usize);

    loop {
        line.clear();
        while i < bytes.len() && bytes[i] != b'\n' {
            line.push(bytes[i]);
            i += 1;
        }
        if i >= bytes.len() {
            return Err("truncated Radiance header".into());
        }
        i += 1; // the newline
        let s = String::from_utf8_lossy(&line).trim().to_string();
        if line.is_empty() || s.is_empty() {
            // Blank line ends the header; the resolution line is next.
            line.clear();
            while i < bytes.len() && bytes[i] != b'\n' {
                line.push(bytes[i]);
                i += 1;
            }
            if i >= bytes.len() {
                return Err("truncated Radiance header".into());
            }
            i += 1;
            let s = String::from_utf8_lossy(&line).trim().to_string();
            let parts: Vec<&str> = s.split_whitespace().collect();
            // Only the standard `-Y h +X w` orientation is accepted: every
            // other flip would need the scanlines reordered, and no HDRI
            // library writes one.
            if parts.len() != 4 || parts[0] != "-Y" || parts[2] != "+X" {
                return Err(format!("unsupported Radiance resolution line: {s:?}"));
            }
            let h: usize = parts[1]
                .parse()
                .map_err(|_| format!("bad height in {s:?}"))?;
            let w: usize = parts[3]
                .parse()
                .map_err(|_| format!("bad width in {s:?}"))?;
            resolution = (w, h);
            break;
        }
        if s.starts_with("#?") {
            saw_magic = true;
        }
        if let Some(fmt) = s.strip_prefix("FORMAT=") {
            if fmt.trim() != "32-bit_rle_rgbe" {
                return Err(format!("unsupported Radiance FORMAT={fmt}"));
            }
        }
    }
    if !saw_magic {
        return Err("not a Radiance file (no #? magic)".into());
    }
    let (w, h) = resolution;
    if w < 2 || h < 1 {
        return Err("environment map too small".into());
    }

    // ── scanlines ──
    let mut pixels = vec![[0.0f32; 3]; w * h];
    let mut row = vec![[0u8; 4]; w];

    for y in 0..h {
        read_scanline(bytes, &mut i, &mut row)?;
        for (x, p) in row.iter().enumerate() {
            pixels[y * w + x] = rgbe_to_rgb(*p);
        }
    }

    EnvMap::new(w, h, pixels)
}

/// One scanline, either encoding, into `row`.
fn read_scanline(bytes: &[u8], i: &mut usize, row: &mut [[u8; 4]]) -> Result<(), String> {
    let w = row.len();
    let need = |i: usize, n: usize| -> Result<(), String> {
        if i + n > bytes.len() {
            Err("truncated Radiance scanline".into())
        } else {
            Ok(())
        }
    };
    need(*i, 4)?;
    let head = [bytes[*i], bytes[*i + 1], bytes[*i + 2], bytes[*i + 3]];
    let rle_len = ((head[2] as usize) << 8) | head[3] as usize;

    if !(head[0] == 2
        && head[1] == 2
        && head[2] & 0x80 == 0
        && rle_len == w
        && (4..=0x7fff).contains(&w))
    {
        // Flat RGBE, one quadruple per pixel. (Old-style RLE — a 1,1,1,n
        // repeat marker — is not produced by any modern writer and is not
        // handled; a file using it fails the pixel count rather than
        // decoding to garbage.)
        need(*i, 4 * w)?;
        for p in row.iter_mut() {
            *p = [bytes[*i], bytes[*i + 1], bytes[*i + 2], bytes[*i + 3]];
            *i += 4;
        }
        return Ok(());
    }
    *i += 4;

    // New-style adaptive RLE: the four components are stored in separate
    // runs, each either a literal block (count <= 128) or a repeat.
    for c in 0..4 {
        let mut x = 0usize;
        while x < w {
            need(*i, 1)?;
            let n = bytes[*i] as usize;
            *i += 1;
            if n > 128 {
                need(*i, 1)?;
                let v = bytes[*i];
                *i += 1;
                let run = n - 128;
                if x + run > w {
                    return Err("Radiance run overruns the scanline".into());
                }
                for _ in 0..run {
                    row[x][c] = v;
                    x += 1;
                }
            } else {
                if n == 0 {
                    return Err("zero-length Radiance run".into());
                }
                need(*i, n)?;
                if x + n > w {
                    return Err("Radiance run overruns the scanline".into());
                }
                for _ in 0..n {
                    row[x][c] = bytes[*i];
                    *i += 1;
                    x += 1;
                }
            }
        }
    }
    Ok(())
}

/// Read a lat-long Radiance `.hdr` file from disk. Native only.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_hdr(path: &std::path::Path) -> Result<EnvMap, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse_hdr(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every built-in must be sampleable and normalised to the same exposure,
    /// or switching environments would silently change image brightness.
    #[test]
    fn builtins_share_an_exposure() {
        for kind in BuiltinEnv::all() {
            let map = generate(kind);
            let mut weighted = 0.0f64;
            let mut weight = 0.0f64;
            let n = 200;
            for j in 0..n {
                let theta = core::f64::consts::PI * (j as f64 + 0.5) / n as f64;
                let (st, ct) = theta.sin_cos();
                for i in 0..n {
                    let phi = core::f64::consts::TAU * (i as f64 + 0.5) / n as f64;
                    let d = Vec3::new(st * phi.cos(), st * phi.sin(), ct);
                    let c = map.radiance(d);
                    let lum = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
                    weighted += lum as f64 * st;
                    weight += st;
                }
            }
            let mean = weighted / weight;
            assert!(
                (mean - BUILTIN_MEAN as f64).abs() < 0.05,
                "{} has mean radiance {mean}, expected ~{BUILTIN_MEAN}",
                kind.name()
            );
        }
    }

    /// The built-ins exist to exercise the importance sampler, so they must
    /// actually be high-frequency: far brighter somewhere than on average.
    #[test]
    fn builtins_are_high_frequency() {
        for kind in BuiltinEnv::all() {
            let map = generate(kind);
            let mut peak = 0.0f32;
            let n = 240;
            for j in 0..n {
                let theta = core::f64::consts::PI * (j as f64 + 0.5) / n as f64;
                let (st, ct) = theta.sin_cos();
                for i in 0..n {
                    let phi = core::f64::consts::TAU * (i as f64 + 0.5) / n as f64;
                    let d = Vec3::new(st * phi.cos(), st * phi.sin(), ct);
                    peak = peak.max(map.radiance(d)[0]);
                }
            }
            assert!(
                peak > 4.0 * BUILTIN_MEAN,
                "{} peak radiance {peak} is too flat to need a CDF",
                kind.name()
            );
        }
    }

    #[test]
    fn names_round_trip() {
        for kind in BuiltinEnv::all() {
            assert_eq!(BuiltinEnv::parse(kind.name()), Some(kind));
        }
        assert_eq!(BuiltinEnv::parse("nope"), None);
    }

    /// A flat-RGBE file, written by hand, decodes to the values that went in.
    #[test]
    fn decodes_a_flat_hdr() {
        let (w, h) = (4usize, 2usize);
        let mut bytes = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 4\n".to_vec();
        for _ in 0..(w * h) {
            // (128,128,128) at exponent 128+1 => (128.5/256)*2 ≈ 1.00390625.
            bytes.extend_from_slice(&[128, 128, 128, 129]);
        }
        let map = parse_hdr(&bytes).expect("decodes");
        let c = map.radiance(Vec3::new(0.0, 0.0, 1.0));
        assert!((c[0] - 1.00390625).abs() < 1e-5, "got {c:?}");
    }

    /// The adaptive RLE path: a repeat run per channel across a wide row.
    #[test]
    fn decodes_new_style_rle() {
        let (w, h) = (8usize, 2usize);
        let mut bytes = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 8\n".to_vec();
        for _ in 0..h {
            bytes.extend_from_slice(&[2, 2, (w >> 8) as u8, (w & 0xFF) as u8]);
            for v in [64u8, 128, 192, 129] {
                bytes.push(128 + w as u8); // a repeat of `w`
                bytes.push(v);
            }
        }
        let map = parse_hdr(&bytes).expect("decodes");
        let c = map.radiance(Vec3::new(0.0, 0.0, 1.0));
        assert!((c[0] - (64.5 / 256.0 * 2.0)).abs() < 1e-5, "got {c:?}");
        assert!((c[2] - (192.5 / 256.0 * 2.0)).abs() < 1e-5, "got {c:?}");
    }

    /// A literal-run scanline, and a mixed one, must decode too.
    #[test]
    fn decodes_literal_runs() {
        let w = 4usize;
        let mut bytes = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 4\n".to_vec();
        bytes.extend_from_slice(&[2, 2, 0, 4]);
        for base in [10u8, 20, 30, 128] {
            bytes.push(w as u8); // literal block of 4
            for k in 0..w {
                bytes.push(base.wrapping_add(k as u8));
            }
        }
        let map = parse_hdr(&bytes).expect("decodes");
        assert_eq!(map.width(), 4);
        assert_eq!(map.height(), 1);
    }

    #[test]
    fn rejects_a_non_radiance_file() {
        let err = parse_hdr(b"PNG\n\n-Y 1 +X 2\n").expect_err("must fail");
        assert!(err.contains("Radiance"), "unhelpful: {err}");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn a_missing_file_is_a_clean_error() {
        let err = load_hdr(std::path::Path::new("/nope/missing.hdr")).expect_err("must fail");
        assert!(err.contains("missing.hdr"), "unhelpful: {err}");
    }
}
