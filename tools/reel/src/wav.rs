//! 24-bit PCM stereo WAV.

use std::io::Write;

pub fn write(path: &std::path::Path, buf: &[[f64; 2]], sr: u32) -> std::io::Result<()> {
    let data = (buf.len() * 6) as u32;
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    w.write_all(b"RIFF")?;
    w.write_all(&(36 + data).to_le_bytes())?;
    w.write_all(b"WAVEfmt ")?;
    w.write_all(&16u32.to_le_bytes())?;
    w.write_all(&1u16.to_le_bytes())?;
    w.write_all(&2u16.to_le_bytes())?;
    w.write_all(&sr.to_le_bytes())?;
    w.write_all(&(sr * 6).to_le_bytes())?;
    w.write_all(&6u16.to_le_bytes())?;
    w.write_all(&24u16.to_le_bytes())?;
    w.write_all(b"data")?;
    w.write_all(&data.to_le_bytes())?;
    for s in buf {
        for &c in s {
            let v = (c.clamp(-1.0, 1.0) * 8_388_607.0).round() as i32;
            w.write_all(&v.to_le_bytes()[..3])?;
        }
    }
    w.flush()
}
