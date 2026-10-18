// Chiang 2016 hair BCSDF in f32: `look::hair::Bcsdf` written out, function
// for function (same names with a `fib_` prefix). Pure functions, no
// bindings: included by the hair kernel and by the skin kernel's strand
// interop. Directions are in the hair frame (x along the fibre, z toward
// the viewer).

const HPI: f32 = 3.14159265358979;

struct Fib {
    v: vec4<f32>,
    s: f32,
    sin2k: vec3<f32>,
    cos2k: vec3<f32>,
    h: f32,
    gamma_o: f32,
    eta: f32,
}

// (eta, beta_m, beta_n, alpha degrees) at offset h.
fn fib_new(shape: vec4<f32>, h_in: f32) -> Fib {
    let bm = shape.y;
    let v0q = 0.726 * bm + 0.812 * bm * bm + 3.7 * pow(bm, 20.0);
    let v0 = v0q * v0q;
    let bn = shape.z;
    let s = sqrt(HPI / 8.0) * (0.265 * bn + 1.194 * bn * bn + 5.372 * pow(bn, 22.0));
    var sk = vec3<f32>(0.0);
    var ck = vec3<f32>(0.0);
    sk.x = sin(radians(shape.w));
    ck.x = sqrt(max(1.0 - sk.x * sk.x, 0.0));
    sk.y = 2.0 * ck.x * sk.x;
    ck.y = ck.x * ck.x - sk.x * sk.x;
    sk.z = 2.0 * ck.y * sk.y;
    ck.z = ck.y * ck.y - sk.y * sk.y;
    let h = clamp(h_in, -0.999999, 0.999999);
    return Fib(vec4<f32>(v0, 0.25 * v0, 4.0 * v0, 4.0 * v0), s, sk, ck, h, asin(h), shape.x);
}

fn fib_i0(x: f32) -> f32 {
    var v = 0.0;
    var x2i = 1.0;
    var ifact = 1.0;
    var i4 = 1.0;
    for (var i = 0; i < 10; i = i + 1) {
        if (i > 1) { ifact = ifact * f32(i); }
        v = v + x2i / (i4 * ifact * ifact);
        x2i = x2i * x * x;
        i4 = i4 * 4.0;
    }
    return v;
}

fn fib_log_i0(x: f32) -> f32 {
    if (x > 12.0) {
        return x + 0.5 * (-log(2.0 * HPI) + log(1.0 / x) + 1.0 / (8.0 * x));
    }
    return log(fib_i0(x));
}

fn fib_mp(cti: f32, cto: f32, sti: f32, sto: f32, v: f32) -> f32 {
    let a = cti * cto / v;
    let b = sti * sto / v;
    if (v <= 0.1) {
        return exp(fib_log_i0(a) - b - 1.0 / v + 0.6931 + log(1.0 / (2.0 * v)));
    }
    return exp(-b) * fib_i0(a) / (sinh(1.0 / v) * 2.0 * v);
}

fn fib_logistic(x_in: f32, s: f32) -> f32 {
    let x = abs(x_in);
    let e = exp(-x / s);
    return e / (s * (1.0 + e) * (1.0 + e));
}

fn fib_logistic_cdf(x: f32, s: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x / s));
}

fn fib_trimmed(x: f32, s: f32) -> f32 {
    return fib_logistic(x, s) / (fib_logistic_cdf(HPI, s) - fib_logistic_cdf(-HPI, s));
}

fn fib_sample_trimmed(u: f32, s: f32) -> f32 {
    let k = fib_logistic_cdf(HPI, s) - fib_logistic_cdf(-HPI, s);
    let x = -s * log(1.0 / (u * k + fib_logistic_cdf(-HPI, s)) - 1.0);
    return clamp(x, -HPI, HPI);
}

fn fib_phi(p: u32, go: f32, gt: f32) -> f32 {
    return 2.0 * f32(p) * gt - 2.0 * go + f32(p) * HPI;
}

fn fib_np(dphi: f32, p: u32, s: f32, go: f32, gt: f32) -> f32 {
    var d = dphi - fib_phi(p, go, gt);
    for (var k = 0; k < 4; k = k + 1) {
        if (d > HPI) { d = d - 2.0 * HPI; }
        if (d < -HPI) { d = d + 2.0 * HPI; }
    }
    return fib_trimmed(d, s);
}

// (sin θ, |cos θ|) of lobe p's tilted cone.
fn fib_tilt(f: Fib, p: u32, sto: f32, cto: f32) -> vec2<f32> {
    if (p == 0u) {
        return vec2<f32>(sto * f.cos2k.y - cto * f.sin2k.y, abs(cto * f.cos2k.y + sto * f.sin2k.y));
    }
    if (p == 1u) {
        return vec2<f32>(sto * f.cos2k.x + cto * f.sin2k.x, abs(cto * f.cos2k.x - sto * f.sin2k.x));
    }
    if (p == 2u) {
        return vec2<f32>(sto * f.cos2k.z + cto * f.sin2k.z, abs(cto * f.cos2k.z - sto * f.sin2k.z));
    }
    return vec2<f32>(sto, abs(cto));
}

fn fib_fresnel(c_in: f32, eta: f32) -> f32 {
    let c = clamp(c_in, 0.0, 1.0);
    let g2 = eta * eta - 1.0 + c * c;
    if (g2 <= 0.0) { return 1.0; }
    let g = sqrt(g2);
    let a = (g - c) / (g + c);
    let b = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
    return 0.5 * a * a * (1.0 + b * b);
}

struct FibGeo {
    sto: f32,
    cto: f32,
    phio: f32,
    gt: f32,
    f: f32,
    l: f32,
}

fn fib_geometry(f: Fib, wo: vec3<f32>) -> FibGeo {
    let sto = wo.x;
    let cto = sqrt(max(1.0 - sto * sto, 0.0));
    let phio = atan2(wo.z, wo.y);
    let stt = sto / f.eta;
    let ctt = sqrt(max(1.0 - stt * stt, 0.0));
    let etap = sqrt(f.eta * f.eta - sto * sto) / max(cto, 1e-12);
    let sgt = f.h / etap;
    let cgt = sqrt(max(1.0 - sgt * sgt, 0.0));
    let gt = asin(clamp(sgt, -1.0, 1.0));
    let fr = fib_fresnel(cto * cos(f.gamma_o), f.eta);
    let l = 2.0 * cgt / max(ctt, 1e-12);
    return FibGeo(sto, cto, phio, gt, fr, l);
}

// The θ-free M_p N_p of the four lobes (`Bcsdf::columns` 0..3).
fn fib_mn(f: Fib, g: FibGeo, wi: vec3<f32>) -> vec4<f32> {
    let sti = wi.x;
    let cti = sqrt(max(1.0 - sti * sti, 0.0));
    let dphi = atan2(wi.z, wi.y) - g.phio;
    var c = vec4<f32>(0.0);
    for (var p = 0u; p < 3u; p = p + 1u) {
        let t = fib_tilt(f, p, g.sto, g.cto);
        c[p] = fib_mp(cti, t.y, sti, t.x, f.v[p]) * fib_np(dphi, p, f.s, f.gamma_o, g.gt);
    }
    c.w = fib_mp(cti, g.cto, sti, g.sto, f.v.w) / (2.0 * HPI);
    return c;
}

// A_p at absorption σ_a (per fibre radius): R, TT, TRT, residual.
fn fib_ap(fr: f32, l: f32, sa: f32) -> vec4<f32> {
    let t = exp(-sa * l);
    let a1 = (1.0 - fr) * (1.0 - fr) * t;
    let a2 = a1 * fr * t;
    return vec4<f32>(fr, a1, a2, a2 * fr * t / (1.0 - t * fr));
}

// `hair_f`: Σ M_p N_p A_p.
fn fib_f(mn: vec4<f32>, fr: f32, l: f32, sa: f32) -> f32 {
    return dot(mn, fib_ap(fr, l, sa));
}

fn fib_pdf(f: Fib, g: FibGeo, wi: vec3<f32>, ap: vec4<f32>) -> f32 {
    let sum = ap.x + ap.y + ap.z + ap.w;
    let sti = wi.x;
    let cti = sqrt(max(1.0 - sti * sti, 0.0));
    let dphi = atan2(wi.z, wi.y) - g.phio;
    var pdf = 0.0;
    for (var p = 0u; p < 3u; p = p + 1u) {
        let t = fib_tilt(f, p, g.sto, g.cto);
        pdf = pdf + fib_mp(cti, t.y, sti, t.x, f.v[p]) * ap[p] / sum * fib_np(dphi, p, f.s, f.gamma_o, g.gt);
    }
    return pdf + fib_mp(cti, g.cto, sti, g.sto, f.v.w) * ap.w / sum / (2.0 * HPI);
}

fn fib_sample(f: Fib, g: FibGeo, ap: vec4<f32>, u: vec4<f32>) -> vec3<f32> {
    let sum = ap.x + ap.y + ap.z + ap.w;
    var p = 3u;
    var acc = 0.0;
    for (var k = 0u; k < 4u; k = k + 1u) {
        acc = acc + ap[k] / sum;
        if (u.x < acc) {
            p = k;
            break;
        }
    }
    let t = fib_tilt(f, p, g.sto, g.cto);
    let v = f.v[p];
    let u1 = max(u.y, 1e-5);
    let ct = 1.0 + v * log(u1 + (1.0 - u1) * exp(-2.0 / v));
    let st = sqrt(max(1.0 - ct * ct, 0.0));
    let cphi = cos(2.0 * HPI * u.z);
    let sti = clamp(-ct * t.x + st * cphi * t.y, -1.0, 1.0);
    let cti = sqrt(max(1.0 - sti * sti, 0.0));
    var dphi = 2.0 * HPI * u.w;
    if (p < 3u) {
        dphi = fib_phi(p, f.gamma_o, g.gt) + fib_sample_trimmed(u.w, f.s);
    }
    let phii = g.phio + dphi;
    return vec3<f32>(sti, cti * cos(phii), cti * sin(phii));
}

// Melanin absorption at λ (`hair::absorb` of the two spectra, θ-weighted).
fn fib_sigma_a(theta: vec2<f32>, lam: f32) -> f32 {
    var w = vec3<f32>(0.0, 0.0, 1.0);
    if (lam <= 450.0) {
        w = vec3<f32>(1.0, 0.0, 0.0);
    } else if (lam <= 550.0) {
        let t = (lam - 450.0) / 100.0;
        w = vec3<f32>(1.0 - t, t, 0.0);
    } else if (lam <= 650.0) {
        let t = (lam - 550.0) / 100.0;
        w = vec3<f32>(0.0, 1.0 - t, t);
    }
    let eu = w.x * 1.37 + w.y * 0.697 + w.z * 0.419;
    let ph = w.x * 1.05 + w.y * 0.4 + w.z * 0.187;
    return theta.x * eu + theta.y * ph;
}
