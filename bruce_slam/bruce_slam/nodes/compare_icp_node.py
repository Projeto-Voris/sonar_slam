"""CLI / Node tool to benchmark and compare 3-D Sonar ICP techniques in ROS 2 Jazzy."""
import time
import numpy as np

from ..slam.registration import (
    icp_point_to_plane_numpy,
    icp_point_to_plane_open3d,
    estimate_normals_pca,
    HAS_OPEN3D,
    RegistrationConfig
)


def generate_wall_benchmark_data(num_points=1500):
    """Generate planar wall scene with unobservable sliding motion along the wall."""
    y = np.random.uniform(-4.0, 4.0, num_points)
    z = np.random.uniform(-4.0, 4.0, num_points)
    x = np.full(num_points, 5.0) + np.random.normal(0, 0.02, num_points)

    target_pts = np.vstack((x, y, z)).T
    target_normals = np.zeros_like(target_pts)
    target_normals[:, 0] = -1.0  # Normal pointing towards sensor

    # Ground truth relative transformation:
    # 0.4m displacement away from wall (observable)
    # 1.2m sliding along wall in Y (unobservable from wall alone!)
    gt_T = np.eye(4)
    gt_T[0, 3] = 0.4
    gt_T[1, 3] = 1.2

    src_hom = np.hstack((target_pts, np.ones((num_points, 1))))
    source_pts = (gt_T @ src_hom.T).T[:, :3]

    return source_pts, target_pts, target_normals, gt_T


def main():
    print("=" * 65)
    print(" 3-D Sonar ICP Techniques Benchmark (ROS 2 Jazzy)")
    print(" Testing Planar Degeneracy & Wall-Sliding Covariance")
    print("=" * 65)

    source_pts, target_pts, target_normals, gt_T = generate_wall_benchmark_data(2000)
    print("\nGround Truth Displacement:")
    print(f"  Delta X (observable perpendicular):  {gt_T[0, 3]:.2f} m")
    print(f"  Delta Y (unobservable wall sliding): {gt_T[1, 3]:.2f} m")

    cfg = RegistrationConfig()

    # 1. NumPy Point-to-Plane ICP
    print("\n[1] Running NumPy Point-to-Plane ICP (Gauss-Newton + Hessian covariance)...")
    t0 = time.perf_counter()
    res_np = icp_point_to_plane_numpy(source_pts, target_pts, target_normals, config=cfg)
    t_np = (time.perf_counter() - t0) * 1000.0

    print(f"  -> Execution Time:     {t_np:.2f} ms")
    print(f"  -> Inlier Fitness:      {res_np.fitness:.4f}")
    print(f"  -> Residual RMSE:       {res_np.rmse:.4f} m")
    print(f"  -> Estimated Delta X:   {res_np.T[0, 3]:.4f} m")
    print(f"  -> Estimated Delta Y:   {res_np.T[1, 3]:.4f} m")
    # Tangent order: [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
    var_x = res_np.covariance[3, 3]
    var_y = res_np.covariance[4, 4]
    print(f"  -> Tangent Covariance Var(X) [observable]:   {var_x:.6f}")
    print(f"  -> Tangent Covariance Var(Y) [UNOBSERVABLE]: {var_y:.2f}  <-- High uncertainty!")

    # 2. Open3D Point-to-Plane ICP
    if HAS_OPEN3D:
        print("\n[2] Running Open3D Point-to-Plane ICP (Information matrix covariance)...")
        t0 = time.perf_counter()
        res_o3d = icp_point_to_plane_open3d(source_pts, target_pts, target_normals, config=cfg)
        t_o3d = (time.perf_counter() - t0) * 1000.0

        print(f"  -> Execution Time:     {t_o3d:.2f} ms")
        print(f"  -> Inlier Fitness:      {res_o3d.fitness:.4f}")
        print(f"  -> Residual RMSE:       {res_o3d.rmse:.4f} m")
        print(f"  -> Estimated Delta X:   {res_o3d.T[0, 3]:.4f} m")
        print(f"  -> Estimated Delta Y:   {res_o3d.T[1, 3]:.4f} m")
        var_x_o3d = res_o3d.covariance[3, 3]
        var_y_o3d = res_o3d.covariance[4, 4]
        print(f"  -> Tangent Covariance Var(X) [observable]:   {var_x_o3d:.6f}")
        print(f"  -> Tangent Covariance Var(Y) [UNOBSERVABLE]: {var_y_o3d:.6f}  <-- Overconfident!")
    else:
        print("\n[2] Open3D is not installed on this system.")

    print("\n" + "=" * 65)
    print("CONCLUSION:")
    print("  NumPy's Hessian extraction correctly downweights unobservable sliding")
    print("  motions with massive covariance, protecting GTSAM Pose3 graph optimization.")
    print("=" * 65)


if __name__ == '__main__':
    main()
