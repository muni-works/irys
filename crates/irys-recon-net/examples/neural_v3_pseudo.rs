//! v3 renderer as teacher, by distillation: the v3 search is the teacher.
//! `neural_v3_pseudo --set v3-ftb --n 120 --out out/neural/data --prefix pseudo-ftb`
//!
//! No label of the photos is read. For each unlabelled photo: the network's
//! scene, the camera's pitch refined on the frame, the appearance fitted,
//! v3's MDL search on the faithful render (net camera), the scale set by the
//! priors (or the gauges); the searched scene is then labelled through the
//! irys TLAS like any synthetic sample and written as a shard, so
//! `neural_train --train pseudo-ftb,train-a --init …` fine-tunes the network
//! toward what the search found.

use std::path::PathBuf;

use irys_recon_net::data::{Sample, hi_path, patches, read_dir, write_shard};
use irys_recon_net::edgejac::EdgeOpts;
use irys_recon_net::faithful::{self, Appearance, Compare3, FitOpts, linearise, local_meshes};
use irys_recon_net::infer::{net_scene, rescale};
use irys_recon_net::render::{resample_rgb8, world_meshes};
use irys_recon_net::scene::{Mat, Object};
use irys_recon_net::search::{State, noise_sigma};
use irys_recon_net::search3::{self, Ctx3};
use irys_recon_net::train::{Infer, Weights};

fn arg<T: std::str::FromStr>(a: &[String], k: &str, d: T) -> T {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(d)
}

const GREY: Mat = Mat { rgb: [0.5; 3], rough: 0.5, metal: false };

fn ctx_of<'a>(gpu: &'static irys_gpu::GpuContext, app: Appearance, target: &'a [[f64; 3]], cam: irys_recon_net::scene::Cam, size: (usize, usize)) -> Ctx3<'a> {
    Ctx3 {
        gpu,
        target,
        size,
        cam,
        app,
        spp: 8,
        seed: 3,
        sigma: noise_sigma(target, size),
        sig_levels: Vec::new(),
        levels: vec![2.0, 1.0, 0.5],
        prior_temp: 0.1,
        decorrelate: true,
        edge: EdgeOpts::default(),
        recolour: true,
        shade: true,
        fd_weak: true,
        tau: 1.0,
        param_sd: Vec::new(),
        pose_sd: 30.0,
        highpass: 2.5,
        exact: None,
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = PathBuf::from(arg(&a, "--data", "out/neural/data".to_string()));
    let wpath = PathBuf::from(arg(&a, "--weights", "out/neural/run2/weights.json".to_string()));
    let set: String = arg(&a, "--set", "v3-ftb".to_string());
    let n: usize = arg(&a, "--n", 120);
    let out = PathBuf::from(arg(&a, "--out", "out/neural/data".to_string()));
    let prefix: String = arg(&a, "--prefix", "pseudo-ftb".to_string());
    let sw: usize = arg(&a, "--size", 256);
    let iters: usize = arg(&a, "--iters", 6);
    let size = (sw, sw * 3 / 4);
    let gpu = irys_diff::gpu::context().expect("gpu");
    let photos: Vec<Sample> = read_dir(&data, &set).into_iter().take(n).collect();
    let w = Weights::load(&wpath).unwrap();
    let mut net = Infer::new(&w, 1);
    let mut outs = Vec::new();
    let t0 = std::time::Instant::now();
    for (k, s) in photos.iter().enumerate() {
        let hi = image::open(hi_path(&data, &set, s.scene.seed)).unwrap().to_rgb8();
        let rgb8: Vec<u8> = resample_rgb8(&hi, size).iter().flat_map(|p| p.map(irys_recon::render::srgb8)).collect();
        let o = net.run(&[patches(&s.rgb)]).pop().unwrap();
        let (objs, alts, cam, gauges, sds) = net_scene(&o, s.scene.cam.hfov, 3, GREY);
        if objs.is_empty() {
            continue;
        }
        let plain: Vec<Object> = objs.iter().map(|x| x.0.clone()).collect();
        // Pitch, then the appearance, then the search.
        let fit = |objs: &[Object], cam| {
            let mut app = Appearance::grey(objs.len());
            let mut cmp = Compare3::new(gpu, objs, &local_meshes(objs), &app, &cam, size).unwrap();
            faithful::fit_appearance(&mut cmp, &cam, &rgb8, &mut app, &FitOpts::default());
            app
        };
        let app0 = fit(&plain, cam);
        let t_a = linearise(&rgb8, app0.gamma);
        let (cam2, _) = search3::refine_pitch(&plain, &ctx_of(gpu, app0, &t_a, cam, size), 6.0);
        let repose = |o: &mut Object| {
            if let Some(px) = cam.project([o.pose[0], o.pose[1], 0.0]) {
                if let Some(xy) = cam2.on_plane(px[0], px[1]) {
                    o.pose[0] = xy[0];
                    o.pose[1] = xy[1];
                }
            }
        };
        let (mut init, mut cands) = (objs.clone(), alts.clone());
        init.iter_mut().for_each(|o| repose(&mut o.0));
        cands.iter_mut().for_each(|v| v.iter_mut().for_each(|o| repose(&mut o.0)));
        let plain: Vec<Object> = init.iter().map(|x| x.0.clone()).collect();
        let app = fit(&plain, cam2);
        let target = linearise(&rgb8, app.gamma);
        let ctx = ctx_of(gpu, app, &target, cam2, size);
        let st0 = State::new(plain.clone());
        let ctx = Ctx3 { sig_levels: search3::noise_levels(&st0, &ctx), ..ctx };
        let ctx = Ctx3 { tau: search3::dispersion(&st0, &ctx), param_sd: sds.clone(), ..ctx };
        let (st, _, _, _) = search3::search(init, &cands, &ctx, iters, 1);
        let mut kept = st.kept();
        let mut c_out = cam2;
        if gauges.is_empty() {
            let sc = search3::prior_scale(&st, &plain, &sds);
            rescale(&mut kept, &mut c_out, sc);
        }
        // The searched scene as a label set, at the net's size.
        let mut scene = s.scene.clone();
        scene.objects = kept;
        scene.cam = c_out.resized((irys_recon_net::data::W, irys_recon_net::data::H));
        scene.clutter = Vec::new();
        let meshes = world_meshes(&scene.objects);
        outs.push(Sample::label(scene, s.rgb.clone(), &meshes));
        println!("[{}/{}] {} objects → {}  {:.0}s", k + 1, photos.len(), plain.len(), outs.last().unwrap().scene.objects.len(), t0.elapsed().as_secs_f64());
        if outs.len() % 20 == 0 || k + 1 == photos.len() {
            write_shard(&out.join(format!("{prefix}-00000000")), &outs).unwrap();
        }
    }
    write_shard(&out.join(format!("{prefix}-00000000")), &outs).unwrap();
}
