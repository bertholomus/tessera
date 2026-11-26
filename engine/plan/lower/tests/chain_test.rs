use tessera_graph::DimName::Rows;
use tessera_graph::{
    Act, DType, Dim, Entry, EntryKind, Graph, Node, NodeId, Op, Output, Placement, Port, Role, Row,
    Shape, Stage, TensorType, Weight,
};
use tessera_lower::sites::Chain;
use tessera_lower::{Device, Tree, Wants};

fn f32_ty(dims: Vec<tessera_graph::DimExpr>) -> TensorType {
    TensorType {
        dtype: DType::F32,
        shape: Shape(dims),
        placement: Placement::Replicated,
    }
}

#[test]
fn chain_stops_turn() {
    let mut g = Graph::new(vec![Dim::bounded(Rows, 8)]);
    let x_ty = f32_ty(vec![Rows.into(), 128u64.into()]);
    let w_ty = f32_ty(vec![128u64.into(), 128u64.into()]);
    let v_ty = f32_ty(vec![128u64.into()]);
    g.weights.push(Weight {
        path: vec![],
        ty: w_ty.clone(),
    });
    g.weights.push(Weight {
        path: vec![],
        ty: v_ty.clone(),
    });
    g.weights.push(Weight {
        path: vec![],
        ty: v_ty.clone(),
    });
    let mut push = |op: Op, args: Vec<NodeId>, ty: TensorType| -> NodeId {
        let id = NodeId(g.nodes.len() as u32);
        g.nodes.push(Node {
            op,
            args,
            ty,
            numerics: None,
        });
        id
    };
    let x = push(Op::Input { entry: 0, port: 0 }, vec![], x_ty.clone());
    let up = push(Op::Input { entry: 0, port: 1 }, vec![], x_ty.clone());
    let w = push(Op::Weight(0), vec![], w_ty);
    let suh = push(Op::Weight(1), vec![], v_ty.clone());
    let svh = push(Op::Weight(2), vec![], v_ty);
    let s1 = push(Op::Row(Row(vec![Stage::Mul])), vec![x, suh], x_ty.clone());
    let h1 = push(
        Op::Row(Row(vec![Stage::Hadamard { block: 128 }])),
        vec![s1],
        x_ty.clone(),
    );
    let mm = push(Op::Matmul, vec![h1, w], x_ty.clone());
    let h2 = push(
        Op::Row(Row(vec![Stage::Hadamard { block: 128 }])),
        vec![mm],
        x_ty.clone(),
    );
    let gate = push(Op::Row(Row(vec![Stage::Mul])), vec![h2, svh], x_ty.clone());
    let act = push(
        Op::Row(Row(vec![Stage::Act(Act::Silu)])),
        vec![gate],
        x_ty.clone(),
    );
    let out = push(Op::Row(Row(vec![Stage::Mul])), vec![act, up], x_ty.clone());
    g.entries.push(Entry {
        kind: EntryKind::Step,
        inputs: vec![
            Port {
                role: Role::Tokens,
                ty: x_ty.clone(),
            },
            Port {
                role: Role::Tokens,
                ty: x_ty,
            },
        ],
        outputs: vec![Output {
            role: Role::Logits,
            node: out,
        }],
    });
    let dev = Device::default();
    let parts = dev.division();
    let tree = Tree::upto(
        &g,
        0,
        Wants::ALL,
        &[(Rows, 1)],
        &[],
        &dev.pins,
        dev.kept,
        &parts,
        0,
    )
    .expect("tree builds");
    let root = tree.outputs[0].0;
    let chain: Chain = tree.site(root).expect("chain matches");
    // Chain stops at gate rather than svh.
    assert_eq!(chain.stages.len(), 2);
    assert_eq!(chain.reads[0], tree.of(gate).unwrap());
}
