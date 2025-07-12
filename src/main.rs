use std::fs::{self, File};
use std::io::Write;
use std::f64::consts::PI;

const OUTPUT_FOLDER: &str = "output";
const DECAY_CONSTANT: f64 = 0.1;
const EXPERT_TELEPORT_FRACTION: f64 = 0.8; // fraction of teleported rank (mass) directed to experts
                    
#[derive(Debug)]
struct Edge {
    source: usize,
    target: usize,
    time_of_creation: usize, // Discrete time
}

fn exponential_decay(t1:usize, t0:usize, weight_at_t0:f64, decay_constant:f64) -> f64 {
    return weight_at_t0 * (- ((t1-t0) as f64) * decay_constant).exp();
} 

fn pagerank_variant(
    edges: &[Edge],
    weights: &[f64],
    num_of_nodes: usize,
    num_of_iterations: usize,
    damping_factor: f64,
    teleportation_targets: &[f64],
) -> Vec<f64> {
    // Rank flow is analogous to mass flow. 
    // Total rank (mass) is conserved.
    // Expert nodes have higher intrinsic rank (mass).

    // Initial uniform rank (mass) distribution over nodes
    let mut rank_values = vec![1.0 / num_of_nodes as f64; num_of_nodes];

    let mut initial_outflow_values = vec![0.0; num_of_nodes];
    for edge in edges {
        initial_outflow_values[edge.source] += 1.0;
    }

    for _ in 0..num_of_iterations {
        // New rank (mass) values are calculated starting with teleportation inflow contribution
        let mut new_rank_values = teleportation_targets
            .iter()
            .map(|&t| (1.0 - damping_factor) * t)
            .collect::<Vec<f64>>();

        // Rank (mass) outflows along edges with speed propotional to edge weights
        let mut outflow_values = vec![0.0; num_of_nodes];
        for (edge, &w) in edges.iter().zip(weights.iter()) {
            outflow_values[edge.source] += w;
            new_rank_values[edge.target] +=
                damping_factor *
                rank_values[edge.source] *
                (w / initial_outflow_values[edge.source]);
        }

        // We redistribute dangling rank (mass) uniformly
        let mut dangling_rank = 0.0;
        for i in 0..num_of_nodes {
            let rank = rank_values[i];
            let initial_outflow = initial_outflow_values[i];
            if initial_outflow > 0.0 {
                let outflow = outflow_values[i];
                let allocated = damping_factor * rank * (outflow / initial_outflow);
                dangling_rank += damping_factor * rank - allocated;
            } else {
                dangling_rank += damping_factor * rank;
            }
        }
        let dangling_share = dangling_rank / num_of_nodes as f64;
        for new_rank in new_rank_values.iter_mut() {
            *new_rank += dangling_share;
        }

        rank_values = new_rank_values;
    }

    // let total_mass: f64 = rank_values.iter().sum();
    // println!("Total rank mass after {} iterations: {}", num_of_iterations, total_mass);
    rank_values
}

fn write_dot(pathname: &str, node_ranks: &[f64], edges: &[Edge], weights: &[f64], experts: &[usize], positions: &[(f64, f64)], current_frame: usize, total_frames: usize, algorithm: &str, decay_desc: &str) {
    let mut file = File::create(pathname).unwrap();
    writeln!(file, "digraph G {{").unwrap();
    writeln!(file, "  nodesep=0.8;").unwrap();
    writeln!(file, "  graph [layout=neato, overlap=false, splines=true, pad=\"1.0,1.0\", fontsize=20];").unwrap();
    writeln!(file, "  labelloc=\"t\";").unwrap();
    writeln!(file, "  labeljust=\"l\";").unwrap();
    writeln!(file, "  labelfontsize=26;").unwrap();
    writeln!(file, "  label=\"Trust flow over time\nAlgorithm: {}\nEdge decay: {}\nFrame: {}/{}\";", algorithm, decay_desc, current_frame, total_frames).unwrap();
    for i in 0..node_ranks.len() {
        let r = node_ranks[i].clamp(0.0, 1.0);
        let level = ((1.0 - r) * 255.0) as u8;
        let fill_color = format!("#{:02X}{:02X}{:02X}", level, level, 255u8);
        let label_text = format!("{} ({:.2})", i, node_ranks[i]);
        let (x, y) = positions[i];
        if experts.contains(&i) {
            writeln!(file,
                "  {} [label=\"{}\", shape=circle, style=filled, fillcolor=\"{}\", color=\"darkgreen\", penwidth=8, fontsize=20, pos=\"{:.2},{:.2}!\", pin=true];",
                i, label_text, fill_color, x, y
            ).unwrap();
        } else {
            writeln!(file,
                "  {} [label=\"{}\", shape=circle, style=filled, fillcolor=\"{}\", fontsize=20, pos=\"{:.2},{:.2}!\", pin=true];",
                i, label_text, fill_color, x, y
            ).unwrap();
        }
    }

    for (e, &w) in edges.iter().zip(weights.iter()) {
        if w == 0.0 {
            writeln!(file,"  {} -> {} [style=invis];", e.source, e.target).unwrap();
        } else {
            let edgewidth = 8.0 * w;
            writeln!(file,"  {} -> {} [penwidth={}];", e.source, e.target, edgewidth).unwrap();
        }
    }

    writeln!(file, "}}").unwrap();
    println!("{pathname} created");
}

fn plot_scenario(name: &str, edges: Vec<Edge>, num_of_nodes: usize, expert_nodes: Vec<usize>) {
    let mut node_positions = Vec::with_capacity(num_of_nodes);
    for i in 0..num_of_nodes {
        let angle = 2.0 * PI * (i as f64) / (num_of_nodes as f64);
        let x = angle.cos();
        let y = angle.sin();
        node_positions.push((x, y));
    }

    let mut teleportation_targets = vec![(1.0 - EXPERT_TELEPORT_FRACTION) / num_of_nodes as f64; num_of_nodes];
    for &e in &expert_nodes {
        teleportation_targets[e] += EXPERT_TELEPORT_FRACTION / expert_nodes.len() as f64;
    }

    let max_time = 20;
    for time in 0..=max_time {
        let decayed_weights: Vec<f64> = edges.iter().map(|e| {
            if e.time_of_creation <= time { exponential_decay(time, e.time_of_creation, 1.0, DECAY_CONSTANT) }
            else { 0.0 }
        }).collect();

        let num_of_iterations = 10;
        let damping_factor = 0.5;
        let ranks = pagerank_variant(
            &edges,
            &decayed_weights,
            num_of_nodes,
            num_of_iterations,
            damping_factor,
            &teleportation_targets,
        );
        let full_folder_pathname = OUTPUT_FOLDER.to_string() + "/" + name;

        fs::create_dir_all(&full_folder_pathname).unwrap();
        let filename = format!("{}/frame_{:03}.dot", &full_folder_pathname, time);
        write_dot(&filename, &ranks, &edges, &decayed_weights, &expert_nodes, &node_positions, time + 1, max_time + 1, "Custom PageRank variant", "Exponential");
    }
}
fn main() {

    {
        let edges = vec![
            Edge { source: 0, target: 1, time_of_creation: 1 },
            Edge { source: 1, target: 2, time_of_creation: 2 },
            Edge { source: 1, target: 3, time_of_creation: 3 },
            Edge { source: 3, target: 4, time_of_creation: 4 },
            Edge { source: 3, target: 5, time_of_creation: 5 },
            Edge { source: 5, target: 1, time_of_creation: 6 },
        ];
        plot_scenario("trust-flow-example", edges, 6, vec![0]); 
    }


}
