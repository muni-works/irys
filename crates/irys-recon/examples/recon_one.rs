use irys_recon::{canon, fit::Target, generate, model::*, render, search::*};
fn main() {
    let seed: u64 = std::env::args().nth(1).unwrap().parse().unwrap();
    let truth = generate::program(seed);
    println!("truth:\n{}\n{}", truth.to_text(), truth.signature());
    let setup = Setup::new(render::views());
    let t = target(&truth, &setup, 64, 1000 + seed).unwrap();
    let bg = background(&setup, 64);
    let tg = Target::new(t, &bg, setup.film.size);
    let r = recover(&tg, &bg, &setup, 64, &mut Algorithmic, &Options::default());
    for s in &r.trace { if s.accepted || std::env::var("ALL").is_ok() { println!("[{}] {} {} -> {:?}", s.round, if s.accepted {"ACC"} else {"   "}, s.why, s.score.map(|x| x.total as i64)); } }
    println!("recovered:\n{}\n{}", r.program.to_text(), r.program.signature());
    println!("topology ok {}", canon::same_topology(&truth, &r.program));
    if let Some(e) = canon::errors(&truth, &r.program) { for (n, v) in e { print!("{n}:{v:.3} "); } println!(); }
    println!("{:.1}s iters {} evals {}", r.seconds, r.iters, r.evals);
    for (n, s) in &r.judge.sigma { print!("{n}±{s:.3} "); } println!();
    println!("weak {:?}", r.judge.weak);
}
