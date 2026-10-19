//! v1's outputs: the program as a vcad document (`.vcad`, vcad's IR DAG,
//! editable in vcad) and as STL (vcad's exporter).

use vcad_ir::{CsgOp, Document, MaterialDef, Node, NodeId, SceneEntry, SketchSegment2D, Vec2, Vec3};
use vcad_kernel::vcad_kernel_sketch::SketchSegment;

use crate::dsl::{Axis, Feature, Program, SEGMENTS};

/// `prog`'s solid as binary STL bytes (vcad's exporter), part frame, mm.
pub fn stl(prog: &Program) -> Result<Vec<u8>, String> {
    let solid = prog.build()?;
    let m = solid.to_mesh(SEGMENTS);
    let spec = vcad_kernel_export::StlSpec {
        name: "irys-recon".into(),
        meshes: vec![vcad_kernel_export::StlMeshSpec { positions: [0, m.vertices.len()], indices: [0, m.indices.len()] }],
    };
    vcad_kernel_export::build_stl(&spec, &m.vertices, &m.indices).map_err(|e| e.to_string())
}

struct Doc {
    d: Document,
    next: NodeId,
}

impl Doc {
    fn add(&mut self, name: &str, op: CsgOp) -> NodeId {
        let id = self.next;
        self.next += 1;
        self.d.nodes.insert(id, Node { id, name: Some(name.into()), op });
        id
    }
    fn cyl(&mut self, name: &str, r: f64, h: f64, x: f64, y: f64, z: f64) -> NodeId {
        let c = self.add(name, CsgOp::Cylinder { radius: r, height: h, segments: SEGMENTS });
        self.add(&format!("{name} at"), CsgOp::Translate { child: c, offset: Vec3::new(x, y, z) })
    }
    fn cut(&mut self, name: &str, a: NodeId, b: NodeId) -> NodeId {
        self.add(name, CsgOp::Difference { left: a, right: b })
    }
}

/// The program as a vcad document: the base outline sketched and
/// extruded, each feature a primitive cut from it (or unioned), in the
/// builder's order; the part's material from the fitted look (`rgb`,
/// `roughness`), named PLA.
pub fn vcad_doc(prog: &Program, rgb: [f64; 3], roughness: f64) -> Document {
    let mut o = Doc { d: Document::new(), next: 1 };
    let p2 = |p: vcad_kernel::vcad_kernel_math::Point2| Vec2::new(p.x, p.y);
    let segs: Vec<SketchSegment2D> = prog
        .outline()
        .into_iter()
        .map(|s| match s {
            SketchSegment::Line { start, end } => SketchSegment2D::Line { start: p2(start), end: p2(end) },
            SketchSegment::Arc { start, end, center, ccw } => SketchSegment2D::Arc { start: p2(start), end: p2(end), center: p2(center), ccw },
        })
        .collect();
    let sk = o.add(
        "base outline",
        CsgOp::Sketch2D { origin: Vec3::new(0.0, 0.0, 0.0), x_dir: Vec3::new(1.0, 0.0, 0.0), y_dir: Vec3::new(0.0, 1.0, 0.0), segments: segs, holes: None },
    );
    let mut cur = o.add("base", CsgOp::Extrude { sketch: sk, direction: Vec3::new(0.0, 0.0, prog.h), twist_angle: None, scale_end: None });
    let h = prog.h;
    for (k, f) in prog.features.iter().enumerate() {
        let n = format!("f{k} {}", f.kind());
        cur = match *f {
            Feature::Counterbore { x, y, r, cr, cd } => {
                let t = o.cyl(&format!("{n} hole"), r, h + 2.0, x, y, -1.0);
                let c = o.cut(&format!("{n} hole cut"), cur, t);
                let b = o.cyl(&format!("{n} bore"), cr, cd + 1.0, x, y, h - cd);
                o.cut(&n, c, b)
            }
            Feature::Slot { x, y, len, r, axis } => {
                let (ux, uy) = if axis == Axis::X { (1.0, 0.0) } else { (0.0, 1.0) };
                let (bx, by) = if axis == Axis::X { (len, 2.0 * r) } else { (2.0 * r, len) };
                let bar = o.add(&format!("{n} bar"), CsgOp::Cube { size: Vec3::new(bx, by, h + 2.0) });
                let bar = o.add(&format!("{n} bar at"), CsgOp::Translate { child: bar, offset: Vec3::new(x - bx / 2.0, y - by / 2.0, -1.0) });
                let c = o.cut(&format!("{n} bar cut"), cur, bar);
                let e1 = o.cyl(&format!("{n} end 1"), r, h + 2.0, x - ux * len / 2.0, y - uy * len / 2.0, -1.0);
                let c = o.cut(&format!("{n} end 1 cut"), c, e1);
                let e2 = o.cyl(&format!("{n} end 2"), r, h + 2.0, x + ux * len / 2.0, y + uy * len / 2.0, -1.0);
                o.cut(&n, c, e2)
            }
            Feature::Boss { x, y, r, bh } => {
                let b = o.cyl(&format!("{n} cylinder"), r, bh + h / 2.0, x, y, h / 2.0);
                o.add(&n, CsgOp::Union { left: cur, right: b })
            }
            _ => {
                let mut c = cur;
                for (i, (x, y, r)) in f.holes().into_iter().enumerate() {
                    let t = o.cyl(&format!("{n} {i}"), r, h + 2.0, x, y, -1.0);
                    c = o.cut(&format!("{n} {i} cut"), c, t);
                }
                c
            }
        };
    }
    let mat = MaterialDef { name: "PLA".into(), color: rgb, metallic: 0.0, roughness, ..MaterialDef::default() };
    o.d.materials.insert("PLA".into(), mat);
    o.d.part_materials.insert("part".into(), "PLA".into());
    o.d.roots.push(SceneEntry { root: cur, material: "PLA".into(), visible: None });
    o.d
}
