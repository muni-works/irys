//! The manifest, the stub manifest, and light → pitch class.
//!
//! Mapping. Light at wavelength λ has frequency c/λ. Transposed down forty
//! octaves (÷ 2⁴⁰) the visible band lands in the treble clef: 760 nm → 359 Hz
//! (F4), 380 nm → 717 Hz (F5). One octave of light is one octave of sound, so
//! the fold onto twelve pitch classes is just `12·log2(f / 440) mod 12`, and
//! A4 = 440 Hz sits at λ = c / (440 · 2⁴⁰) ≈ 619.7 nm, orange. Red (700 nm) is
//! G, yellow (580 nm) A#, green (530 nm) C, cyan (490 nm) C#, blue (450 nm) D#,
//! violet (400 nm) F. The forty-octave transposition is the classic one (it puts
//! the sodium D line on A#4) and it is the only free choice: no scale is imposed.
//!
//! Chroma is measured relative to an equal-energy white, so a flat spectrum
//! folds to a flat chroma: the red end spans more nanometres per semitone than
//! the blue end, and without the white reference every frame would lean red.

use serde::{Deserialize, Serialize};

pub const C_LIGHT: f64 = 299_792_458.0;
pub const OCTAVES_DOWN: f64 = 40.0;

/// Pitch class (0 = C … 11 = B, fractional) of wavelength `nm`.
pub fn pitch_class(nm: f64) -> f64 {
    let hz = C_LIGHT / (nm * 1e-9) / 2f64.powf(OCTAVES_DOWN);
    let midi = 69.0 + 12.0 * (hz / 440.0).log2();
    midi.rem_euclid(12.0)
}

/// The real manifest's per-sequence spectrum block.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SpectrumBlock {
    pub bin_centres_nm: Vec<f64>,
    pub per_frame: Vec<Vec<f64>>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Sequence {
    #[serde(alias = "name")]
    pub id: String,
    pub path: String,
    #[serde(alias = "frame_count")]
    pub frames: usize,
    pub fps: f64,
    #[serde(default)]
    pub caption: String,
    /// Bin centres (nm); falls back to the manifest's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wavelengths_nm: Option<Vec<f64>>,
    /// One mean spectrum per frame, radiance per bin.
    #[serde(alias = "mean_spectra", default)]
    pub spectra: Vec<Vec<f64>>,
    /// `{bin_centres_nm, per_frame}`, as `examples/reel.rs` writes it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spectrum: Option<SpectrumBlock>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Manifest {
    #[serde(default)]
    pub stub: bool,
    #[serde(default)]
    pub note: String,
    #[serde(default)]
    pub wavelengths_nm: Vec<f64>,
    pub sequences: Vec<Sequence>,
}

impl Manifest {
    /// Fold the `spectrum` block form into `spectra` / `wavelengths_nm`.
    pub fn normalise(mut self) -> Self {
        for s in &mut self.sequences {
            if let Some(b) = s.spectrum.take() {
                if s.spectra.is_empty() {
                    s.spectra = b.per_frame;
                }
                s.wavelengths_nm.get_or_insert(b.bin_centres_nm);
            }
        }
        self
    }
    pub fn get(&self, id: &str) -> Option<&Sequence> {
        self.sequences.iter().find(|s| s.id == id)
    }
    pub fn bins<'a>(&'a self, s: &'a Sequence) -> &'a [f64] {
        s.wavelengths_nm.as_deref().unwrap_or(&self.wavelengths_nm)
    }
}

/// Fold a spectrum onto twelve pitch classes, linear between neighbours.
fn fold(bins: &[f64], spec: &[f64]) -> [f64; 12] {
    let mut c = [0.0; 12];
    for (&nm, &v) in bins.iter().zip(spec) {
        let pc = pitch_class(nm);
        let lo = pc.floor() as usize % 12;
        let fr = pc - pc.floor();
        c[lo] += v.max(0.0) * (1.0 - fr);
        c[(lo + 1) % 12] += v.max(0.0) * fr;
    }
    c
}

/// Chroma relative to equal-energy white, normalised to max 1.
pub fn chroma(bins: &[f64], spec: &[f64]) -> [f64; 12] {
    let white = fold(bins, &vec![1.0; bins.len()]);
    let raw = fold(bins, spec);
    let mut c = [0.0; 12];
    for i in 0..12 {
        c[i] = if white[i] > 0.0 { raw[i] / white[i] } else { 0.0 };
    }
    let m = c.iter().cloned().fold(0.0, f64::max);
    if m > 0.0 {
        c.iter_mut().for_each(|v| *v /= m);
    }
    c
}

fn planck(nm: f64, t: f64) -> f64 {
    let l = nm * 1e-9;
    let (h, k) = (6.626e-34, 1.381e-23);
    1.0 / (l.powi(5) * ((h * C_LIGHT / (l * k * t)).exp() - 1.0))
}

fn sig(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn gauss(x: f64, mu: f64, w: f64) -> f64 {
    (-((x - mu) / w).powi(2) / 2.0).exp()
}

/// A plausible stand-in for the real manifest: illuminant × reflectance for
/// each of the reel's looks, with a little per-frame drift, and the thin film
/// sweeping its thickness so its interference peak walks across the band.
pub fn stub() -> Manifest {
    let bins: Vec<f64> = (0..41).map(|i| 380.0 + 10.0 * i as f64).collect();
    let frames = 48;
    let fps = 24.0;
    type Look = fn(f64, f64) -> f64;
    let looks: [(&str, &str, Look); 6] = [
        ("gold", "gold under daylight", |nm, u| planck(nm, 5600.0) * (0.22 + 0.72 * sig((nm - 515.0 - 10.0 * u) / 16.0))),
        ("sky", "the sky dome, Rayleigh", |nm, u| planck(nm, 5800.0 + 600.0 * u) * (nm / 550.0).powf(-4.0)),
        ("skin", "skin, melanin and blood", |nm, u| {
            planck(nm, 5000.0)
                * (0.2 + 0.45 * sig((nm - 590.0) / 14.0) - (0.06 + 0.03 * u) * (gauss(nm, 542.0, 9.0) + gauss(nm, 577.0, 9.0)))
        }),
        ("thin_film", "a soap film thinning", |nm, u| {
            let d = 620.0 - 380.0 * u; // nm
            planck(nm, 6000.0) * (0.04 + (2.0 * std::f64::consts::PI * 1.33 * d / nm).cos().powi(2) * 0.96)
        }),
        ("classroom", "the classroom, tungsten and oak", |nm, u| planck(nm, 3000.0 + 300.0 * u) * (0.15 + 0.5 * sig((nm - 565.0) / 25.0))),
        ("pavilion", "the stone pavilion, overcast", |nm, u| planck(nm, 6500.0) * (0.45 + 0.1 * (nm - 380.0) / 400.0 + 0.04 * u)),
    ];
    let sequences = looks
        .iter()
        .map(|(id, cap, f)| {
            let spectra = (0..frames)
                .map(|k| {
                    let u = k as f64 / (frames - 1) as f64;
                    let row: Vec<f64> = bins.iter().map(|&nm| f(nm, u).max(0.0)).collect();
                    let m = row.iter().cloned().fold(0.0, f64::max);
                    row.iter().map(|v| (v / m * 1e4).round() / 1e4).collect()
                })
                .collect();
            Sequence {
                id: id.to_string(),
                path: format!("out/stub-frames/{id}/%05d.png"),
                frames,
                fps,
                caption: format!("STUB: {cap}"),
                wavelengths_nm: None,
                spectra,
                spectrum: None,
            }
        })
        .collect();
    Manifest {
        stub: true,
        note: "STUB. Synthesised illuminant x reflectance spectra, not irys renders. \
               Replace with /Users/cam/Developer/irys-assets/reel/manifest.json."
            .into(),
        wavelengths_nm: bins,
        sequences,
    }
}
