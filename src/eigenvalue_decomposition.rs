// This code is based on EigenvalueDecomposition.java from the library Jama which is release to public domain.
// The solver in the Eigen C++ library is also based on the code from Jama.

/** Eigenvalues and eigenvectors of a real matrix.
<P>
    If A is symmetric, then A = V*D*V' where the eigenvalue matrix D is
    diagonal and the eigenvector matrix V is orthogonal.
    I.e. A = V.times(D.times(V.transpose())) and
    V.times(V.transpose()) equals the identity matrix.
<P>
    If A is not symmetric, then the eigenvalue matrix D is block diagonal
    with the real eigenvalues in 1-by-1 blocks and any complex eigenvalues,
    lambda + i*mu, in 2-by-2 blocks, [lambda, mu; -mu, lambda].  The
    columns of V represent the eigenvectors in the sense that A*V = V*D,
    i.e. A.times(V) equals V.times(D).  The matrix V may be badly
    conditioned, or even singular, so the validity of the equation
    A = V*D*inverse(V) depends upon V.cond().
**/
use nalgebra::{self, DMatrix, DVector};

pub struct EigenvalueDecomposition {
    /* ------------------------
      Class variables
    * ------------------------ */
    /** Row and column dimension (square matrix).
    @serial matrix dimension.
    */
    n: usize,

    /** Symmetry flag.
    @serial internal symmetry flag.
    */
    // issymmetric: bool,

    /** Arrays for internal storage of eigenvalues.
    @serial internal storage of eigenvalues.
    */
    d: DVector<f64>,
    e: DVector<f64>,

    /** Array for internal storage of eigenvectors.
    @serial internal storage of eigenvectors.
    */
    V: DMatrix<f64>,

    /** Array for internal storage of nonsymmetric Hessenberg form.
    @serial internal storage of nonsymmetric Hessenberg form.
    */
    H: DMatrix<f64>,

    /** Working storage for nonsymmetric algorithm.
    @serial working storage for nonsymmetric algorithm.
    */
    ort: DVector<f64>,
}

/* ------------------------
  Private Methods
* ------------------------ */

// Symmetric Householder reduction to tridiagonal form.

impl EigenvalueDecomposition {
    fn tred2(&mut self) {
        let Self { d, e, V, .. } = self;
        let n = self.n;

        //  This is derived from the Algol procedures tred2 by
        //  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
        //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
        //  Fortran subroutine in EISPACK.

        for j in 0..n {
            d[j] = V[(n - 1, j)];
        }

        // Householder reduction to tridiagonal form.

        for i in (n - 1)..0 {
            // Scale to avoid under/overflow.

            let mut scale = 0.0;
            let mut h = 0.0;
            for k in 0..i {
                scale = scale + d[k].abs();
            }
            if scale == 0.0 {
                e[i] = d[i - 1];
                for j in 0..i {
                    d[j] = V[(i - 1, j)];
                    V[(i, j)] = 0.0;
                    V[(j, i)] = 0.0;
                }
            } else {
                // Generate Householder vector.

                for k in 0..i {
                    d[k] /= scale;
                    h += d[k] * d[k];
                }
                let mut f = d[i - 1];
                let mut g = h.sqrt();
                if f > 0.0 {
                    g = -g;
                }
                e[i] = scale * g;
                h = h - f * g;
                d[i - 1] = f - g;
                for j in 0..i {
                    e[j] = 0.0;
                }

                // Apply similarity transformation to remaining columns.

                for j in 0..i {
                    f = d[j];
                    V[(j, i)] = f;
                    g = e[j] + V[(j, j)] * f;
                    for k in (j + 1)..i {
                        g += V[(k, j)] * d[k];
                        e[k] += V[(k, j)] * f;
                    }
                    e[j] = g;
                }
                f = 0.0;
                for j in 0..i {
                    e[j] /= h;
                    f += e[j] * d[j];
                }
                let hh = f / (h + h);
                for j in 0..i {
                    e[j] -= hh * d[j];
                }
                for j in 0..i {
                    f = d[j];
                    g = e[j];
                    for k in j..i {
                        V[(k, j)] -= f * e[k] + g * d[k];
                    }
                    d[j] = V[(i - 1, j)];
                    V[(i, j)] = 0.0;
                }
            }
            d[i] = h;
        }

        // Accumulate transformations.

        for i in 0..(n - 1) {
            V[(n - 1, i)] = V[(i, i)];
            V[(i, i)] = 1.0;
            let h = d[i + 1];
            if h != 0.0 {
                for k in 0..=i {
                    d[k] = V[(k, i + 1)] / h;
                }
                for j in 0..=i {
                    let mut g = 0.0;
                    for k in 0..=i {
                        g += V[(k, i + 1)] * V[(k, j)];
                    }
                    for k in 0..=i {
                        V[(k, j)] -= g * d[k];
                    }
                }
            }
            for k in 0..=i {
                V[(k, i + 1)] = 0.0;
            }
        }
        for j in 0..n {
            d[j] = V[(n - 1, j)];
            V[(n - 1, j)] = 0.0;
        }
        V[(n - 1, n - 1)] = 1.0;
        e[0] = 0.0;
    }

    // Symmetric tridiagonal QL algorithm.

    fn tql2(&mut self) {
        //  This is derived from the Algol procedures tql2, by
        //  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
        //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
        //  Fortran subroutine in EISPACK.

        let Self { d, e, V, .. } = self;
        let n = self.n;

        for i in 1..n {
            e[i - 1] = e[i];
        }
        e[n - 1] = 0.0;

        let mut f = 0.0;
        let mut tst1: f64 = 0.0;
        let eps = (2.0_f64).powf(-52.0);
        for l in 0..n {
            // Find small subdiagonal element

            tst1 = tst1.max(d[l].abs() + e[l].abs());
            let mut m = l;
            while m < n {
                if e[m].abs() <= eps * tst1 {
                    break;
                }
                m += 1;
            }

            // If m == l, d[l] is an eigenvalue,
            // otherwise, iterate.

            if m > l {
                // let mut iter = 0;
                loop {
                    // iter += 1; // (Could check iteration count here.)

                    // Compute implicit shift

                    let mut g = d[l];
                    let mut p = (d[l + 1] - g) / (2.0 * e[l]);
                    let mut r = p.hypot(1.0);
                    if p < 0.0 {
                        r = -r;
                    }
                    d[l] = e[l] / (p + r);
                    d[l + 1] = e[l] * (p + r);
                    let dl1 = d[l + 1];
                    let mut h = g - d[l];
                    for i in (l + 2)..n {
                        d[i] -= h;
                    }
                    f = f + h;

                    // Implicit QL transformation.

                    p = d[m];
                    let mut c = 1.0;
                    let mut c2 = c;
                    let mut c3 = c;
                    let el1 = e[l + 1];
                    let mut s = 0.0;
                    let mut s2 = 0.0;
                    for i in (l..=(m - 1)).rev() {
                        c3 = c2;
                        c2 = c;
                        s2 = s;
                        g = c * e[i];
                        h = c * p;
                        r = p.hypot(e[i]);
                        e[i + 1] = s * r;
                        s = e[i] / r;
                        c = p / r;
                        p = c * d[i] - s * g;
                        d[i + 1] = h + s * (c * g + s * d[i]);

                        // Accumulate transformation.

                        for k in 0..n {
                            h = V[(k, i + 1)];
                            V[(k, i + 1)] = s * V[(k, i)] + c * h;
                            V[(k, i)] = c * V[(k, i)] - s * h;
                        }
                    }
                    p = -s * s2 * c3 * el1 * e[l] / dl1;
                    e[l] = s * p;
                    d[l] = c * p;

                    // Check for convergence.
                    if !(e[l].abs() > eps * tst1) {
                        break;
                    }
                }
            }
            d[l] = d[l] + f;
            e[l] = 0.0;
        }

        // Sort eigenvalues and corresponding vectors.

        for i in 0..(n - 1) {
            let mut k = i;
            let mut p = d[i];
            for j in (i + 1)..n {
                if d[j] < p {
                    k = j;
                    p = d[j];
                }
            }
            if k != i {
                d[k] = d[i];
                d[i] = p;
                for j in 0..n {
                    p = V[(j, i)];
                    V[(j, i)] = V[(j, k)];
                    V[(j, k)] = p;
                }
            }
        }
    }

    // Nonsymmetric reduction to Hessenberg form.

    fn orthes(&mut self) {
        //  This is derived from the Algol procedures orthes and ortran,
        //  by Martin and Wilkinson, Handbook for Auto. Comp.,
        //  Vol.ii-Linear Algebra, and the corresponding
        //  Fortran subroutines in EISPACK.
        let Self { V, H, ort, .. } = self;
        let n = self.n;

        let low = 0;
        let high = n - 1;

        for m in (low + 1)..(high - 1) {
            // Scale column.

            let mut scale = 0.0;
            for i in m..high {
                scale = scale + H[(i, m - 1)].abs();
            }
            if scale != 0.0 {
                // Compute Householder transformation.

                let mut h = 0.0;
                for i in (m..=high).rev() {
                    ort[i] = H[(i, m - 1)] / scale;
                    h += ort[i] * ort[i];
                }
                let mut g = h.sqrt();
                if ort[m] > 0.0 {
                    g = -g;
                }
                h = h - ort[m] * g;
                ort[m] = ort[m] - g;

                // Apply Householder similarity transformation
                // H = (I-u*u'/h)*H*(I-u*u')/h)

                for j in m..n {
                    let mut f = 0.0;
                    for i in (m..=high).rev() {
                        f += ort[i] * H[(i, j)];
                    }
                    f = f / h;
                    for i in m..=high {
                        H[(i, j)] -= f * ort[i];
                    }
                }

                for i in 0..=high {
                    let mut f = 0.0;
                    for j in (m..=high).rev() {
                        f += ort[j] * H[(i, j)];
                    }
                    f = f / h;
                    for j in m..=high {
                        H[(i, j)] -= f * ort[j];
                    }
                }
                ort[m] = scale * ort[m];
                H[(m, m - 1)] = scale * g;
            }
        }

        // Accumulate transformations (Algol's ortran).

        for i in 0..n {
            for j in 0..n {
                V[(i, j)] = if i == j { 1.0 } else { 0.0 };
            }
        }

        for m in ((low + 1)..(high - 1)).rev() {
            if H[(m, m - 1)] != 0.0 {
                for i in (m + 1)..high {
                    ort[i] = H[(i, m - 1)];
                }
                for j in m..=high {
                    let mut g = 0.0;
                    for i in m..=high {
                        g += ort[i] * V[(i, j)];
                    }
                    // Double division avoids possible underflow
                    g = (g / ort[m]) / H[(m, m - 1)];
                    for i in m..=high {
                        V[(i, j)] += g * ort[i];
                    }
                }
            }
        }
    }

    // Complex scalar division.
    fn cdiv(xr: f64, xi: f64, yr: f64, yi: f64) -> (f64, f64) {
        if yr.abs() > yi.abs() {
            let r = yi / yr;
            let d = yr + r * yi;
            ((xr + r * xi) / d, (xi - r * xr) / d)
        } else {
            let r = yr / yi;
            let d = yi + r * yr;
            ((r * xr + xi) / d, (r * xi - xr) / d)
        }
    }

    // Nonsymmetric reduction from Hessenberg to real Schur form.

    fn hqr2(&mut self) {
        //  This is derived from the Algol procedure hqr2,
        //  by Martin and Wilkinson, Handbook for Auto. Comp.,
        //  Vol.ii-Linear Algebra, and the corresponding
        //  Fortran subroutine in EISPACK.

        let Self { e, d, V, H, .. } = self;

        // Initialize
        let nn = self.n;
        let mut n = nn - 1;
        let low = 0;
        let high = nn - 1;
        let eps = (2.0_f64).powf(-52.0);
        let mut exshift = 0.0;
        let mut p = 0.0;
        let mut q = 0.0;
        let mut r = 0.0;
        let mut s = 0.0;
        let mut z = 0.0;
        let mut t;
        let mut w;
        let mut x;
        let mut y;

        // Store roots isolated by balanc and compute matrix norm

        let mut norm = 0.0;
        for i in 0..nn {
            if (i as isize) < (low as isize) || (i as isize) > (high as isize) {
                let i = i as usize;
                d[i] = H[(i, i)];
                e[i] = 0.0;
            }
            for j in (1.max(i) - 1)..nn {
                let i = i as usize;
                let j = j as usize;
                norm = norm + H[(i, j)].abs();
            }
        }

        // Outer loop over eigenvalue index

        let mut iter = 0;
        // while n >= low {
        'outer: loop {
            // Look for single small sub-diagonal element

            let mut l = n;
            while l as isize > low as isize {
                s = H[(l - 1, l - 1)].abs() + H[(l, l)].abs();
                if s == 0.0 {
                    s = norm;
                }
                if H[(l, l - 1)].abs() < eps * s {
                    break;
                }
                l -= 1;
            }

            // Check for convergence
            // One root found

            if l == n {
                H[(n, n)] = H[(n, n)] + exshift;
                d[n] = H[(n, n)];
                e[n] = 0.0;
                n -= 1;
                iter = 0;

            // Two roots found
            } else if (l as isize) == (n as isize) - 1 {
                w = H[(n, n - 1)] * H[(n - 1, n)];
                p = (H[(n - 1, n - 1)] - H[(n, n)]) / 2.0;
                q = p * p + w;
                z = q.abs().sqrt();
                H[(n, n)] = H[(n, n)] + exshift;
                H[(n - 1, n - 1)] = H[(n - 1, n - 1)] + exshift;
                x = H[(n, n)];

                // Real pair

                if q >= 0.0 {
                    if p >= 0.0 {
                        z = p + z;
                    } else {
                        z = p - z;
                    }
                    d[n - 1] = x + z;
                    d[n] = d[n - 1];
                    if z != 0.0 {
                        d[n] = x - w / z;
                    }
                    e[n - 1] = 0.0;
                    e[n] = 0.0;
                    x = H[(n, n - 1)];
                    s = x.abs() + z.abs();
                    p = x / s;
                    q = z / s;
                    r = (p * p + q * q).sqrt();
                    p = p / r;
                    q = q / r;

                    // Row modification

                    for j in (n - 1)..nn {
                        z = H[(n - 1, j)];
                        H[(n - 1, j)] = q * z + p * H[(n, j)];
                        H[(n, j)] = q * H[(n, j)] - p * z;
                    }

                    // Column modification

                    for i in 0..=n {
                        z = H[(i, n - 1)];
                        H[(i, n - 1)] = q * z + p * H[(i, n)];
                        H[(i, n)] = q * H[(i, n)] - p * z;
                    }

                    // Accumulate transformations

                    for i in low..=high {
                        z = V[(i, n - 1)];
                        V[(i, n - 1)] = q * z + p * V[(i, n)];
                        V[(i, n)] = q * V[(i, n)] - p * z;
                    }

                // Complex pair
                } else {
                    d[n - 1] = x + p;
                    d[n] = x + p;
                    e[n - 1] = z;
                    e[n] = -z;
                }
                iter = 0;
                if n >= low + 2 {
                    n = n - 2;
                } else {
                    break 'outer;
                }

            // No convergence yet
            } else {
                // Form shift

                x = H[(n, n)];
                y = 0.0;
                w = 0.0;
                if (l as isize) < (n as isize) {
                    y = H[(n - 1, n - 1)];
                    w = H[(n, n - 1)] * H[(n - 1, n)];
                }

                // Wilkinson's original ad hoc shift

                if iter == 10 {
                    exshift += x;
                    for i in low..=n {
                        H[(i, i)] -= x;
                    }
                    s = H[(n, n - 1)].abs() + H[(n - 1, n - 2)].abs();
                    x = 0.75 * s;
                    y = x;
                    w = -0.4375 * s * s;
                }

                // MATLAB's new ad hoc shift

                if iter == 30 {
                    s = (y - x) / 2.0;
                    s = s * s + w;
                    if s > 0.0 {
                        s = s.sqrt();
                        if y < x {
                            s = -s;
                        }
                        s = x - w / ((y - x) / 2.0 + s);
                        for i in low..=n {
                            H[(i, i)] -= s;
                        }
                        exshift += s;
                        x = 0.964;
                        y = x;
                        w = x;
                    }
                }

                iter = iter + 1; // (Could check iteration count here.)

                // Look for two consecutive small sub-diagonal elements

                let mut m = n - 2;
                while (m as isize) >= (l as isize) {
                    z = H[(m, m)];
                    r = x - z;
                    s = y - z;
                    p = (r * s - w) / H[(m + 1, m)] + H[(m, m + 1)];
                    q = H[(m + 1, m + 1)] - z - r - s;
                    r = H[(m + 2, m + 1)];
                    s = p.abs() + q.abs() + r.abs();
                    p = p / s;
                    q = q / s;
                    r = r / s;
                    if m == l {
                        break;
                    }
                    if H[(m, m - 1)].abs() * (q.abs() + r.abs())
                        < eps
                            * (p.abs()
                                * (H[(m - 1, m - 1)].abs() + z.abs() + H[(m + 1, m + 1)].abs()))
                    {
                        break;
                    }
                    m -= 1;
                }

                for i in (m + 2)..=n {
                    H[(i, i - 2)] = 0.0;
                    if i > m + 2 {
                        H[(i, i - 3)] = 0.0;
                    }
                }

                // Double QR step involving rows l:n and columns m:n

                for k in m..=(n - 1) {
                    let notlast = k != n - 1;
                    if k != m {
                        p = H[(k, k - 1)];
                        q = H[(k + 1, k - 1)];
                        r = if notlast { H[(k + 2, k - 1)] } else { 0.0 };
                        x = p.abs() + q.abs() + r.abs();
                        if x == 0.0 {
                            continue;
                        }
                        p = p / x;
                        q = q / x;
                        r = r / x;
                    }

                    s = (p * p + q * q + r * r).sqrt();
                    if p < 0.0 {
                        s = -s;
                    }
                    if s != 0.0 {
                        if k != m {
                            H[(k, k - 1)] = -s * x;
                        } else if l != m {
                            H[(k, k - 1)] = -H[(k, k - 1)];
                        }
                        p = p + s;
                        x = p / s;
                        y = q / s;
                        z = r / s;
                        q = q / p;
                        r = r / p;

                        // Row modification

                        for j in k..nn {
                            p = H[(k, j)] + q * H[(k + 1, j)];
                            if notlast {
                                p = p + r * H[(k + 2, j)];
                                H[(k + 2, j)] = H[(k + 2, j)] - p * z;
                            }
                            H[(k, j)] = H[(k, j)] - p * x;
                            H[(k + 1, j)] = H[(k + 1, j)] - p * y;
                        }

                        // Column modification

                        for i in 0..=n.min(k + 3) {
                            p = x * H[(i, k)] + y * H[(i, k + 1)];
                            if notlast {
                                p = p + z * H[(i, k + 2)];
                                H[(i, k + 2)] = H[(i, k + 2)] - p * r;
                            }
                            H[(i, k)] = H[(i, k)] - p;
                            H[(i, k + 1)] = H[(i, k + 1)] - p * q;
                        }

                        // Accumulate transformations

                        for i in low..=high {
                            p = x * V[(i, k)] + y * V[(i, k + 1)];
                            if notlast {
                                p = p + z * V[(i, k + 2)];
                                V[(i, k + 2)] = V[(i, k + 2)] - p * r;
                            }
                            V[(i, k)] = V[(i, k)] - p;
                            V[(i, k + 1)] = V[(i, k + 1)] - p * q;
                        }
                    } // (s != 0)
                } // k loop
            } // check convergence
        } // while (n >= low)

        // Backsubstitute to find vectors of upper triangular form

        if norm == 0.0 {
            return;
        }

        //   for (n = nn-1; n >= 0; n--) {
        for n in (0..(nn - 1)).rev() {
            p = d[n];
            q = e[n];

            // Real vector

            if q == 0.0 {
                let mut l = n;
                H[(n, n)] = 1.0;
                // for i = n-1; i >= 0; i--) {
                for i in (0..=(n - 1)).rev() {
                    w = H[(i, i)] - p;
                    r = 0.0;
                    for j in l..=n {
                        r = r + H[(i, j)] * H[(j, n)];
                    }
                    if e[i] < 0.0 {
                        z = w;
                        s = r;
                    } else {
                        l = i;
                        if e[i] == 0.0 {
                            if w != 0.0 {
                                H[(i, n)] = -r / w;
                            } else {
                                H[(i, n)] = -r / (eps * norm);
                            }

                        // Solve real equations
                        } else {
                            x = H[(i, i + 1)];
                            y = H[(i + 1, i)];
                            q = (d[i] - p) * (d[i] - p) + e[i] * e[i];
                            t = (x * s - z * r) / q;
                            H[(i, n)] = t;
                            if x.abs() > z.abs() {
                                H[(i + 1, n)] = (-r - w * t) / x;
                            } else {
                                H[(i + 1, n)] = (-s - y * t) / z;
                            }
                        }

                        // Overflow control

                        t = H[(i, n)].abs();
                        if (eps * t) * t > 1.0 {
                            for j in i..=n {
                                H[(j, n)] = H[(j, n)] / t;
                            }
                        }
                    }
                }

            // Complex vector
            } else if q < 0.0 {
                let mut l = n - 1;

                // Last vector component imaginary so matrix is triangular

                if H[(n, n - 1)].abs() > H[(n - 1, n)].abs() {
                    H[(n - 1, n - 1)] = q / H[(n, n - 1)];
                    H[(n - 1, n)] = -(H[(n, n)] - p) / H[(n, n - 1)];
                } else {
                    let (cdivr, cdivi) = Self::cdiv(0.0, -H[(n - 1, n)], H[(n - 1, n - 1)] - p, q);
                    H[(n - 1, n - 1)] = cdivr;
                    H[(n - 1, n)] = cdivi;
                }
                H[(n, n - 1)] = 0.0;
                H[(n, n)] = 1.0;
                // for i = n-2; i >= 0; i--) {
                for i in (0..(n - 1)).rev() {
                    let mut ra = 0.0;
                    let mut sa = 0.0;
                    let mut vr;
                    let vi;
                    for j in l..=n {
                        ra = ra + H[(i, j)] * H[(j, n - 1)];
                        sa = sa + H[(i, j)] * H[(j, n)];
                    }
                    w = H[(i, i)] - p;

                    if e[i] < 0.0 {
                        z = w;
                        r = ra;
                        s = sa;
                    } else {
                        l = i;
                        if e[i] == 0.0 {
                            let (cdivr, cdivi) = Self::cdiv(-ra, -sa, w, q);
                            H[(i, n - 1)] = cdivr;
                            H[(i, n)] = cdivi;
                        } else {
                            // Solve complex equations

                            x = H[(i, i + 1)];
                            y = H[(i + 1, i)];
                            vr = (d[i] - p) * (d[i] - p) + e[i] * e[i] - q * q;
                            vi = (d[i] - p) * 2.0 * q;
                            if vr == 0.0 && vi == 0.0 {
                                vr = eps * norm * (w.abs() + q.abs() + x.abs() + y.abs() + z.abs());
                            }
                            let (cdivr, cdivi) = Self::cdiv(
                                x * r - z * ra + q * sa,
                                x * s - z * sa - q * ra,
                                vr,
                                vi,
                            );
                            H[(i, n - 1)] = cdivr;
                            H[(i, n)] = cdivi;
                            if x.abs() > (z.abs() + q.abs()) {
                                H[(i + 1, n - 1)] = (-ra - w * H[(i, n - 1)] + q * H[(i, n)]) / x;
                                H[(i + 1, n)] = (-sa - w * H[(i, n)] - q * H[(i, n - 1)]) / x;
                            } else {
                                let (cdivr, cdivi) =
                                    Self::cdiv(-r - y * H[(i, n - 1)], -s - y * H[(i, n)], z, q);
                                H[(i + 1, n - 1)] = cdivr;
                                H[(i + 1, n)] = cdivi;
                            }
                        }

                        // Overflow control

                        t = H[(i, n - 1)].abs().min(H[(i, n)].abs());
                        if (eps * t) * t > 1.0 {
                            for j in i..=n {
                                H[(j, n - 1)] = H[(j, n - 1)] / t;
                                H[(j, n)] = H[(j, n)] / t;
                            }
                        }
                    }
                }
            }
        }

        // Vectors of isolated roots

        for i in 0..nn {
            if i < low || i > high {
                for j in i..nn {
                    V[(i, j)] = H[(i, j)];
                }
            }
        }

        // Back transformation to get eigenvectors of original matrix

        //   for j = nn-1; j >= low; j--) {
        for j in (low..nn).rev() {
            for i in low..=high {
                z = 0.0;
                for k in low..=j.min(high) {
                    z = z + V[(i, k)] * H[(k, j)];
                }
                V[(i, j)] = z;
            }
        }
    }

    /* ------------------------
      Constructor
    * ------------------------ */

    /** Check for symmetry, then construct the eigenvalue decomposition
        Structure to access D and V.
    @param Arg    Square matrix
    */

    pub fn new(A: DMatrix<f64>) -> Self {
        let n = A.ncols();
        let mut issymmetric = true;
        'outer: for j in 0..n {
            for i in 0..n {
                if A[(i, j)] != A[(j, i)] {
                    issymmetric = false;
                    break 'outer;
                }
            }
        }
        let mut ret = Self {
            n,
            V: A.clone(),
            d: DVector::zeros(n),
            e: DVector::zeros(n),
            H: A.clone(), // TODO: Don't allocate this if it isn't needed (symmetric case)
            ort: DVector::zeros(n), // TODO: Don't allocate this if it isn't needed (symmetric case)
        };

        if issymmetric {
            // Tridiagonalize.
            ret.tred2();

            // Diagonalize.
            ret.tql2();
        } else {
            // Reduce to Hessenberg form.
            ret.orthes();

            // Reduce Hessenberg to real Schur form.
            ret.hqr2();
        }
        ret
    }

    /* ------------------------
      Public Methods
    * ------------------------ */

    /** Return the eigenvector matrix
    @return     V
    */

    pub fn get_eigenvectors(&self) -> &DMatrix<f64> {
        &self.V
    }

    /** Return the real parts of the eigenvalues
    @return     real(diag(D))
    */

    pub fn get_real_eigenvalues(&self) -> &DVector<f64> {
        &self.d
    }

    /** Return the imaginary parts of the eigenvalues
    @return     imag(diag(D))
    */

    pub fn get_imag_eigenvalues(&self) -> &DVector<f64> {
        &self.e
    }

    /** Return the block diagonal eigenvalue matrix
    @return     D
    */

    pub fn get_eigenvalue_matrix(&self) -> DMatrix<f64> {
        let n = self.n;
        let Self { d, e, .. } = self;
        let mut D = DMatrix::zeros(n, n);
        for i in 0..n {
            D[(i, i)] = d[i];
            if e[i] > 0.0 {
                D[(i, i + 1)] = e[i];
            } else if e[i] < 0.0 {
                D[(i, i - 1)] = e[i];
            }
        }
        D
    }
}
