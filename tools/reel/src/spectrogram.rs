//! A log-frequency spectrogram PNG of the mix, section boundaries in white.

use std::f64::consts::PI;

fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * PI / len as f64;
        for s in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (wr, wi) = ((ang * k as f64).cos(), (ang * k as f64).sin());
                let (a, b) = (s + k, s + k + len / 2);
                let (xr, xi) = (re[b] * wr - im[b] * wi, re[b] * wi + im[b] * wr);
                re[b] = re[a] - xr;
                im[b] = im[a] - xi;
                re[a] += xr;
                im[a] += xi;
            }
        }
        len <<= 1;
    }
}

fn colour(v: f64) -> [u8; 3] {
    // black → indigo → magenta → orange → pale yellow
    let stops = [[0.0, 0.0, 0.02], [0.2, 0.05, 0.4], [0.7, 0.15, 0.45], [0.98, 0.55, 0.1], [1.0, 1.0, 0.75]];
    let x = v.clamp(0.0, 1.0) * 4.0;
    let i = (x.floor() as usize).min(3);
    let f = x - i as f64;
    let c: Vec<u8> = (0..3).map(|k| ((stops[i][k] * (1.0 - f) + stops[i + 1][k] * f) * 255.0) as u8).collect();
    [c[0], c[1], c[2]]
}

pub fn write(path: &std::path::Path, buf: &[[f64; 2]], sr: f64, marks: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
    let (w, h, nfft) = (1600usize, 480usize, 4096usize);
    let hop = (buf.len().saturating_sub(nfft)) / w;
    let win: Vec<f64> = (0..nfft).map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / nfft as f64).cos()).collect();
    let (fmin, fmax) = (30.0f64, 20_000.0f64);
    let mut db = vec![0.0f64; w * h];
    for x in 0..w {
        let mut re: Vec<f64> = (0..nfft).map(|i| 0.5 * (buf[x * hop + i][0] + buf[x * hop + i][1]) * win[i]).collect();
        let mut im = vec![0.0; nfft];
        fft(&mut re, &mut im);
        for y in 0..h {
            let f = fmin * (fmax / fmin).powf(1.0 - y as f64 / (h - 1) as f64);
            let k = ((f / sr * nfft as f64) as usize).clamp(1, nfft / 2 - 1);
            let m = (re[k] * re[k] + im[k] * im[k]).sqrt();
            db[y * w + x] = 20.0 * (m + 1e-9).log10();
        }
    }
    let top = db.iter().cloned().fold(f64::MIN, f64::max);
    let mut img = vec![0u8; w * h * 3];
    for (i, &d) in db.iter().enumerate() {
        let c = colour((d - (top - 90.0)) / 90.0);
        img[i * 3..i * 3 + 3].copy_from_slice(&c);
    }
    for &m in marks {
        let x = ((m * w as f64) as usize).min(w - 1);
        for y in (0..h).step_by(2) {
            img[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&[255, 255, 255]);
        }
    }
    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut enc = png::Encoder::new(file, w as u32, h as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()?.write_image_data(&img)?;
    Ok(())
}
