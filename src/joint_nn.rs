use kiddo::ImmutableKdTree;
use kiddo::SquaredEuclidean;
use std::num::NonZero;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JointNearestBackend {
    KdTree,
    DualTree,
}

impl JointNearestBackend {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "kdtree" | "kd-tree" => Some(Self::KdTree),
            "dual-tree" | "dualtree" => Some(Self::DualTree),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::KdTree => "kdtree",
            Self::DualTree => "dual-tree",
        }
    }
}

pub fn calc_joint_nn<const K: usize>(
    coordinates: [&[f64]; K],
    backend: JointNearestBackend,
) -> Result<f64, String> {
    let points = build_points(coordinates)?;
    match backend {
        JointNearestBackend::KdTree => calc_joint_nn_kdtree(&points),
        JointNearestBackend::DualTree => calc_joint_nn_dual_tree(&points),
    }
}

fn build_points<const K: usize>(coordinates: [&[f64]; K]) -> Result<Vec<[f64; K]>, String> {
    let points_len = coordinates[0].len();
    if points_len < 2 {
        return Err(format!(
            "need at least two points for {K}D nearest neighbor"
        ));
    }
    if coordinates
        .iter()
        .any(|coordinate| coordinate.len() != points_len)
    {
        return Err(format!(
            "all coordinate series must have equal length for {K}D nearest neighbor"
        ));
    }

    let mut points = Vec::with_capacity(points_len);
    for frame_idx in 0..points_len {
        let mut point = [0.0; K];
        for dimension_idx in 0..K {
            point[dimension_idx] = coordinates[dimension_idx][frame_idx];
        }
        points.push(point);
    }
    Ok(points)
}

fn calc_joint_nn_kdtree<const K: usize>(points: &[[f64; K]]) -> Result<f64, String> {
    let points_len = points.len();
    let kdtree: ImmutableKdTree<f64, K> = ImmutableKdTree::new_from_slice(points);
    let mut distance_total = 0.0;
    for point in points {
        let neighbor_count = if points_len < 8 { points_len } else { 8 };
        let result = kdtree
            .nearest_n::<SquaredEuclidean>(point, NonZero::new(neighbor_count).unwrap())
            .into_iter()
            .skip(1)
            .map(|neighbor| neighbor.distance)
            .find(|distance| *distance > 0.0)
            .ok_or_else(|| {
                format!("need at least two distinct points for {K}D nearest neighbor")
            })?;
        distance_total += result.sqrt().ln();
    }
    Ok(distance_total)
}

fn calc_joint_nn_dual_tree<const K: usize>(points: &[[f64; K]]) -> Result<f64, String> {
    let (unique_points, multiplicities) = compress_duplicate_points(points);
    if unique_points.len() < 2 {
        return Err(format!(
            "need at least two distinct points for {K}D nearest neighbor"
        ));
    }

    let tree = DualTree::new(unique_points);
    let mut best_squared_distances = vec![f64::INFINITY; tree.points.len()];
    traverse_symmetric_pair(&tree, tree.root, tree.root, &mut best_squared_distances);

    best_squared_distances.iter().zip(multiplicities).try_fold(
        0.0,
        |acc, (distance, multiplicity)| {
            if !distance.is_finite() || *distance <= 0.0 {
                return Err(format!(
                    "need at least two distinct points for {K}D nearest neighbor"
                ));
            }
            Ok(acc + multiplicity as f64 * distance.sqrt().ln())
        },
    )
}

fn compress_duplicate_points<const K: usize>(points: &[[f64; K]]) -> (Vec<[f64; K]>, Vec<usize>) {
    let mut sorted = points.to_vec();
    sorted.sort_by(compare_points);

    let mut unique_points = Vec::new();
    let mut multiplicities = Vec::new();
    for point in sorted {
        if unique_points
            .last()
            .is_some_and(|previous| points_are_equal(previous, &point))
        {
            *multiplicities.last_mut().expect("multiplicity exists") += 1;
        } else {
            unique_points.push(point);
            multiplicities.push(1);
        }
    }
    (unique_points, multiplicities)
}

fn compare_points<const K: usize>(left: &[f64; K], right: &[f64; K]) -> std::cmp::Ordering {
    for dimension in 0..K {
        let ordering = left[dimension].total_cmp(&right[dimension]);
        if ordering != std::cmp::Ordering::Equal {
            return ordering;
        }
    }
    std::cmp::Ordering::Equal
}

fn points_are_equal<const K: usize>(left: &[f64; K], right: &[f64; K]) -> bool {
    (0..K).all(|dimension| left[dimension] == right[dimension])
}

struct DualTree<const K: usize> {
    points: Vec<[f64; K]>,
    point_indices: Vec<usize>,
    nodes: Vec<Node<K>>,
    root: usize,
}

struct Node<const K: usize> {
    start: usize,
    end: usize,
    bounds_min: [f64; K],
    bounds_max: [f64; K],
    left: Option<usize>,
    right: Option<usize>,
}

impl<const K: usize> Node<K> {
    fn is_leaf(&self) -> bool {
        self.left.is_none()
    }

    fn len(&self) -> usize {
        self.end - self.start
    }
}

impl<const K: usize> DualTree<K> {
    const LEAF_SIZE: usize = 16;

    fn new(points: Vec<[f64; K]>) -> Self {
        let mut tree = Self {
            point_indices: (0..points.len()).collect(),
            points,
            nodes: Vec::new(),
            root: 0,
        };
        tree.root = tree.build_node(0, tree.point_indices.len());
        tree
    }

    fn build_node(&mut self, start: usize, end: usize) -> usize {
        let (bounds_min, bounds_max) = self.bounds_for_range(start, end);
        let node_id = self.nodes.len();
        self.nodes.push(Node {
            start,
            end,
            bounds_min,
            bounds_max,
            left: None,
            right: None,
        });

        if end - start > Self::LEAF_SIZE {
            let split_dimension = widest_dimension(bounds_min, bounds_max);
            self.point_indices[start..end].sort_by(|left, right| {
                self.points[*left][split_dimension].total_cmp(&self.points[*right][split_dimension])
            });
            let mid = start + (end - start) / 2;
            let left = self.build_node(start, mid);
            let right = self.build_node(mid, end);
            self.nodes[node_id].left = Some(left);
            self.nodes[node_id].right = Some(right);
        }

        node_id
    }

    fn bounds_for_range(&self, start: usize, end: usize) -> ([f64; K], [f64; K]) {
        let first = self.points[self.point_indices[start]];
        let mut bounds_min = first;
        let mut bounds_max = first;

        for &point_index in &self.point_indices[(start + 1)..end] {
            let point = self.points[point_index];
            for dimension in 0..K {
                bounds_min[dimension] = bounds_min[dimension].min(point[dimension]);
                bounds_max[dimension] = bounds_max[dimension].max(point[dimension]);
            }
        }

        (bounds_min, bounds_max)
    }
}

fn widest_dimension<const K: usize>(bounds_min: [f64; K], bounds_max: [f64; K]) -> usize {
    let mut best_dimension = 0;
    let mut best_width = bounds_max[0] - bounds_min[0];
    for dimension in 1..K {
        let width = bounds_max[dimension] - bounds_min[dimension];
        if width > best_width {
            best_dimension = dimension;
            best_width = width;
        }
    }
    best_dimension
}

fn traverse_symmetric_pair<const K: usize>(
    tree: &DualTree<K>,
    left_id: usize,
    right_id: usize,
    best_squared_distances: &mut [f64],
) {
    let left = &tree.nodes[left_id];
    let right = &tree.nodes[right_id];

    if left_id != right_id {
        let bound = minimum_squared_distance_between_nodes(left, right);
        let max_best = max_best_squared_distance(tree, left, best_squared_distances).max(
            max_best_squared_distance(tree, right, best_squared_distances),
        );
        if bound >= max_best {
            return;
        }
    }

    match (left.is_leaf(), right.is_leaf()) {
        (true, true) => update_leaf_pair(tree, left_id, right_id, best_squared_distances),
        (false, false) if left_id == right_id => {
            let left_child = left.left.expect("internal node has left child");
            let right_child = left.right.expect("internal node has right child");
            traverse_symmetric_pair(tree, left_child, left_child, best_squared_distances);
            traverse_symmetric_pair(tree, left_child, right_child, best_squared_distances);
            traverse_symmetric_pair(tree, right_child, right_child, best_squared_distances);
        }
        (false, true) => {
            let left_child = left.left.expect("internal node has left child");
            let right_child = left.right.expect("internal node has right child");
            traverse_symmetric_pair(tree, left_child, right_id, best_squared_distances);
            traverse_symmetric_pair(tree, right_child, right_id, best_squared_distances);
        }
        (true, false) => {
            let left_child = right.left.expect("internal node has left child");
            let right_child = right.right.expect("internal node has right child");
            traverse_symmetric_pair(tree, left_id, left_child, best_squared_distances);
            traverse_symmetric_pair(tree, left_id, right_child, best_squared_distances);
        }
        (false, false) => {
            let split_left = left.len() >= right.len();
            if split_left {
                let left_child = left.left.expect("internal node has left child");
                let right_child = left.right.expect("internal node has right child");
                traverse_symmetric_pair(tree, left_child, right_id, best_squared_distances);
                traverse_symmetric_pair(tree, right_child, right_id, best_squared_distances);
            } else {
                let left_child = right.left.expect("internal node has left child");
                let right_child = right.right.expect("internal node has right child");
                traverse_symmetric_pair(tree, left_id, left_child, best_squared_distances);
                traverse_symmetric_pair(tree, left_id, right_child, best_squared_distances);
            }
        }
    }
}

fn update_leaf_pair<const K: usize>(
    tree: &DualTree<K>,
    left_id: usize,
    right_id: usize,
    best_squared_distances: &mut [f64],
) {
    let left = &tree.nodes[left_id];
    let right = &tree.nodes[right_id];

    for left_offset in left.start..left.end {
        let left_point_index = tree.point_indices[left_offset];
        let right_start = if left_id == right_id {
            left_offset + 1
        } else {
            right.start
        };
        for right_offset in right_start..right.end {
            let right_point_index = tree.point_indices[right_offset];
            let distance = squared_distance(
                &tree.points[left_point_index],
                &tree.points[right_point_index],
            );
            if distance <= 0.0 {
                continue;
            }
            if distance < best_squared_distances[left_point_index] {
                best_squared_distances[left_point_index] = distance;
            }
            if distance < best_squared_distances[right_point_index] {
                best_squared_distances[right_point_index] = distance;
            }
        }
    }
}

fn max_best_squared_distance<const K: usize>(
    tree: &DualTree<K>,
    node: &Node<K>,
    best_squared_distances: &[f64],
) -> f64 {
    tree.point_indices[node.start..node.end]
        .iter()
        .map(|point_index| best_squared_distances[*point_index])
        .fold(0.0, f64::max)
}

fn minimum_squared_distance_between_nodes<const K: usize>(left: &Node<K>, right: &Node<K>) -> f64 {
    let mut distance = 0.0;
    for dimension in 0..K {
        if left.bounds_max[dimension] < right.bounds_min[dimension] {
            distance += (right.bounds_min[dimension] - left.bounds_max[dimension]).powi(2);
        } else if right.bounds_max[dimension] < left.bounds_min[dimension] {
            distance += (left.bounds_min[dimension] - right.bounds_max[dimension]).powi(2);
        }
    }
    distance
}

fn squared_distance<const K: usize>(left: &[f64; K], right: &[f64; K]) -> f64 {
    let mut distance = 0.0;
    for dimension in 0..K {
        distance += (left[dimension] - right[dimension]).powi(2);
    }
    distance
}
