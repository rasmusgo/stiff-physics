use hyperdual::{Float, Hyperdual};
use nalgebra::{self, DMatrix, DVector, Point, SVector};

trait OneHot {
    type H;
    fn from_one_hot(index: usize) -> Self::H;
}

impl<T: hyperdual::Zero + hyperdual::One + Copy + nalgebra::Scalar, const N: usize> OneHot
    for Hyperdual<T, N>
{
    type H = Hyperdual<T, N>;
    /// Create a new dual number from a real number and set the real value or a derivative to one.
    ///
    /// All other parts are set to zero.
    #[inline]
    fn from_one_hot(index: usize) -> Hyperdual<T, N>
    where
        T: hyperdual::Zero,
    {
        let mut dual = Hyperdual::<T, N>::from_real(T::zero());
        dual[index] = T::one();
        dual
    }
}

#[cfg_attr(feature = "persistence", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone)]
pub struct Spring {
    pub p1: usize,
    pub p2: usize,
    pub length: f64,
    pub k: f64,
    pub d: f64,
}

pub const D: usize = 2;

fn spring_force<const S: usize>(
    p1_pos: SVector<Hyperdual<f64, S>, D>,
    p1_vel: SVector<Hyperdual<f64, S>, D>,
    p2_pos: SVector<Hyperdual<f64, S>, D>,
    p2_vel: SVector<Hyperdual<f64, S>, D>,
    relaxed_length: f64,
    k: f64, // Spring constant
    d: f64,
) -> SVector<Hyperdual<f64, S>, D> {
    let dpos = p2_pos - p1_pos;
    let dvel = p2_vel - p1_vel;
    let spring_length = dpos.dot(&dpos).sqrt();
    let spring_dir = dpos / spring_length;
    let force_magnitude: Hyperdual<f64, S> = Hyperdual::from_real(k)
        * (spring_length - Hyperdual::from_real(relaxed_length))
        + Hyperdual::from_real(d) * spring_dir.dot(&dvel);
    spring_dir * -force_magnitude
}

pub fn create_diff_eq_system(
    points: &[Point<f64, D>],
    point_masses: &[f64],
    springs: &[Spring],
) -> (DMatrix<f64>, DVector<f64>) {
    assert_eq!(points.len(), point_masses.len());
    let num_points = points.len();
    let block_size = num_points * D;
    let system_size = block_size * 2 + 1;

    // Initial state
    let mut y0 = DVector::zeros(system_size);
    for i in 0..num_points {
        for j in 0..D {
            y0[i * D + j] = points[i][j] as f64;
        }
    }
    y0[system_size - 1] = 1.0; // For transformation to homogenous system.

    // Dual numbers for automatic differentiation of springs. (Spatial derivatives, not time derivatives).
    let mut p1_pos = SVector::<Hyperdual<f64, 9>, D>::new(
        Hyperdual::from_one_hot(1),
        Hyperdual::from_one_hot(2),
    );
    let mut p2_pos = SVector::<Hyperdual<f64, 9>, D>::new(
        Hyperdual::from_one_hot(3),
        Hyperdual::from_one_hot(4),
    );
    let mut p1_vel = SVector::<Hyperdual<f64, 9>, D>::new(
        Hyperdual::from_one_hot(5),
        Hyperdual::from_one_hot(6),
    );
    let mut p2_vel = SVector::<Hyperdual<f64, 9>, D>::new(
        Hyperdual::from_one_hot(7),
        Hyperdual::from_one_hot(8),
    );

    // Construct A matrix for y' = Ay. (Time derivative of state vector).
    let mut mat_a = DMatrix::zeros(system_size, system_size);

    // Equations for variable substitutions
    for i in 0..num_points * D {
        // "velocity is velocity"
        mat_a[(i, block_size + i)] = 1.0;
    }

    // Equations for spring forces
    for spring in springs {
        let p1_loc = spring.p1 * D;
        let p2_loc = spring.p2 * D;

        // Set parameters to spring function.
        p1_pos[0][0] = y0[p1_loc];
        p1_pos[1][0] = y0[p1_loc + 1];
        p2_pos[0][0] = y0[p2_loc];
        p2_pos[1][0] = y0[p2_loc + 1];
        p1_vel[0][0] = y0[block_size + p1_loc];
        p1_vel[1][0] = y0[block_size + p1_loc + 1];
        p2_vel[0][0] = y0[block_size + p2_loc];
        p2_vel[1][0] = y0[block_size + p2_loc + 1];

        let p1_mass = point_masses[spring.p1];
        let p2_mass = point_masses[spring.p2];

        let force = spring_force(
            p1_pos,
            p1_vel,
            p2_pos,
            p2_vel,
            spring.length,
            spring.k,
            spring.d,
        );

        for j in 0..D {
            for k in 0..D {
                // Acceleration based on position
                mat_a[(block_size + p1_loc + j, p1_loc + k)] -= force[j][1 + k] / p1_mass; // p1 acc from pos of p1.
                mat_a[(block_size + p1_loc + j, p2_loc + k)] -= force[j][3 + k] / p1_mass; // p1 acc from pos of p2.
                mat_a[(block_size + p2_loc + j, p1_loc + k)] += force[j][1 + k] / p2_mass; // p2 acc from pos of p1.
                mat_a[(block_size + p2_loc + j, p2_loc + k)] += force[j][3 + k] / p2_mass; // p2 acc from pos of p2.

                // Damping
                mat_a[(block_size + p1_loc + j, block_size + p1_loc + k)] -=
                    force[j][5 + k] / p1_mass; // p1 acc from vel of p1.
                mat_a[(block_size + p1_loc + j, block_size + p2_loc + k)] -=
                    force[j][7 + k] / p1_mass; // p1 acc from vel of p2.
                mat_a[(block_size + p2_loc + j, block_size + p1_loc + k)] +=
                    force[j][5 + k] / p2_mass; // p2 acc from vel of p1.
                mat_a[(block_size + p2_loc + j, block_size + p2_loc + k)] +=
                    force[j][7 + k] / p2_mass; // p2 acc from vel of p2.
            }
            // Offset for linearization around y0.
            let mut constant_term = force[j][0];
            for k in 0..D {
                constant_term -= force[j][1 + k] * y0[p1_loc + k]
                    + force[j][3 + k] * y0[p2_loc + k]
                    + force[j][5 + k] * y0[block_size + p1_loc + k]
                    + force[j][7 + k] * y0[block_size + p2_loc + k];
            }

            // Constant acceleration term.
            mat_a[(block_size + p1_loc + j, system_size - 1)] -= constant_term / p1_mass;
            mat_a[(block_size + p2_loc + j, system_size - 1)] += constant_term / p2_mass;
        }
    }

    (mat_a, y0)
}
