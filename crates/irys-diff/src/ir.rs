//! One way to run a tensor-IR graph: compiled C (f32), or the interpreter
//! (f64).
//!
//! A trimmed `kosm::ir::Program`. Natively a graph is [`TensorGraph::to_c`],
//! built by the system C compiler (`$TANG_CC`, default `cc`) and `dlopen`ed;
//! where there is no compiler, or under `IRYS_IR=interp`, it is the IR's own
//! reference interpreter. Both run the same graph, so a test holds one
//! against the other. tang-gpu's `CpuProgram` is the same loader, but
//! tang-gpu links wgpu 25 and irys is on 30, so the forty lines live here.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

use tang_compile::tensor::c::CProgram;
use tang_compile::tensor::{Format, Op, TensorGraph, TensorId, numel};

type RunFn = unsafe extern "C" fn();
type PtrFn = unsafe extern "C" fn(i32) -> *mut core::ffi::c_void;

struct Native {
    program: CProgram,
    _lib: libloading::Library,
    run: RunFn,
    input: PtrFn,
    output: PtrFn,
    dir: PathBuf,
}

impl Drop for Native {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

enum Backend {
    Native(Native),
    Interp {
        g: TensorGraph,
        inputs: Vec<Vec<f64>>,
        out: Vec<Vec<f64>>,
    },
}

/// A graph, lowered and loaded, with its inputs resident.
pub struct Program {
    backend: Backend,
    outputs: Vec<TensorId>,
    input_lens: Vec<usize>,
}

/// Whether `IRYS_IR=interp` asks for the interpreter everywhere.
pub fn force_interp() -> bool {
    std::env::var("IRYS_IR").is_ok_and(|v| v == "interp")
}

impl Program {
    /// Native when a C compiler is available (and `IRYS_IR` allows it),
    /// the interpreter otherwise. Inputs start at zero.
    pub fn compile(g: &TensorGraph, outputs: &[TensorId]) -> Self {
        if !force_interp() {
            match Self::native(g, outputs) {
                Ok(p) => return p,
                Err(e) => eprintln!("irys-diff: no native program ({e}); using the interpreter"),
            }
        }
        Self::interpreted(g, outputs)
    }

    /// The reference interpreter, always: f64 throughout.
    pub fn interpreted(g: &TensorGraph, outputs: &[TensorId]) -> Self {
        let input_lens = input_lens(g);
        let inputs = input_lens.iter().map(|&n| vec![0.0; n]).collect();
        Self {
            backend: Backend::Interp {
                g: g.clone(),
                inputs,
                out: Vec::new(),
            },
            outputs: outputs.to_vec(),
            input_lens,
        }
    }

    /// The compiled C program, or why there is none.
    pub fn native(g: &TensorGraph, outputs: &[TensorId]) -> Result<Self, String> {
        static SEQ: AtomicUsize = AtomicUsize::new(0);
        let program = g.to_c(outputs);
        if program
            .program
            .buffers
            .iter()
            .any(|b| b.format != Format::F32)
        {
            return Err("only f32 buffers are supported".into());
        }
        let dir = std::env::temp_dir().join(format!(
            "irys-diff-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let src = dir.join("prog.c");
        let lib_path = dir.join("prog.so");
        std::fs::write(&src, &program.source).map_err(|e| e.to_string())?;
        let cc = std::env::var("TANG_CC").unwrap_or_else(|_| "cc".into());
        let o = Command::new(&cc)
            .args(["-O2", "-ffp-contract=off", "-shared", "-fPIC", "-o"])
            .arg(&lib_path)
            .arg(&src)
            .arg("-lm")
            .output()
            .map_err(|e| format!("{cc}: {e}"))?;
        if !o.status.success() {
            return Err(String::from_utf8_lossy(&o.stderr).into_owned());
        }
        // SAFETY: the library is the unit just generated; tang's C backend
        // declares these three symbols with these signatures.
        let native = unsafe {
            let lib = libloading::Library::new(&lib_path).map_err(|e| e.to_string())?;
            let run = *lib.get::<RunFn>(b"tang_run\0").map_err(|e| e.to_string())?;
            let input = *lib
                .get::<PtrFn>(b"tang_input\0")
                .map_err(|e| e.to_string())?;
            let output = *lib
                .get::<PtrFn>(b"tang_output\0")
                .map_err(|e| e.to_string())?;
            Native {
                program,
                _lib: lib,
                run,
                input,
                output,
                dir,
            }
        };
        Ok(Self {
            backend: Backend::Native(native),
            outputs: outputs.to_vec(),
            input_lens: input_lens(g),
        })
    }

    /// Whether this is the compiled C program rather than the interpreter.
    pub fn is_native(&self) -> bool {
        matches!(self.backend, Backend::Native(_))
    }

    /// Overwrite graph input `k` (rounded to f32 natively).
    pub fn write(&mut self, k: usize, data: &[f64]) {
        assert_eq!(data.len(), self.input_lens[k], "input {k} length");
        match &mut self.backend {
            Backend::Native(p) => {
                let v: Vec<f32> = data.iter().map(|&x| x as f32).collect();
                // SAFETY: `tang_input(k)` is a static f32 array of this length.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        v.as_ptr(),
                        (p.input)(k as i32) as *mut f32,
                        v.len(),
                    )
                }
            }
            Backend::Interp { inputs, .. } => inputs[k].copy_from_slice(data),
        }
    }

    /// Run every kernel once.
    pub fn run(&mut self) {
        match &mut self.backend {
            // SAFETY: `tang_run` touches only the library's own buffers.
            Backend::Native(p) => unsafe { (p.run)() },
            Backend::Interp { g, inputs, out } => *out = g.eval(inputs, &self.outputs),
        }
    }

    /// Output `i` (in `outputs` order) after the last [`run`](Self::run).
    pub fn read(&self, i: usize) -> Vec<f64> {
        match &self.backend {
            Backend::Native(p) => {
                let len = p.program.program.buffers[p.program.outputs[i]].len;
                // SAFETY: as for `write`.
                let s =
                    unsafe { std::slice::from_raw_parts((p.output)(i as i32) as *const f32, len) };
                s.iter().map(|&v| v as f64).collect()
            }
            Backend::Interp { out, .. } => out[i].clone(),
        }
    }
}

fn input_lens(g: &TensorGraph) -> Vec<usize> {
    g.nodes()
        .iter()
        .filter(|n| matches!(n.op, Op::Input(_)))
        .map(|n| numel(&n.shape))
        .collect()
}
