import numpy as np
import torch
from pyquaternion import Quaternion
from mmdet.datasets.builder import PIPELINES
from mmdet3d.core.points import BasePoints
import cv2 as cv
import torch.nn.functional as F
CNT = 0

def build_lidar2ego(results, device=None, dtype=torch.float32):
    """Build 4x4 transform matrix: lidar -> ego (lidar timestamp ego)."""
    curr = results["curr"]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = Quaternion(curr["lidar2ego_rotation"]).rotation_matrix
    T[:3, 3] = np.array(curr["lidar2ego_translation"], dtype=np.float32)
    T = torch.from_numpy(T)
    if device is not None:
        T = T.to(device)
    return T.to(dtype)

def transform_xyz_rowvec(xyz, T):
    """xyz: (N,3) row vectors. T: (4,4) lidar->ego.
    Return: (N,3) in ego.
    """
    R = T[:3, :3]
    t = T[:3, 3]
    return xyz.matmul(R.T) + t.unsqueeze(0)


def points_to_ego(points: BasePoints, results):
    """Transform BasePoints from lidar frame to ego frame (only xyz)."""
    pts = points.tensor
    T = build_lidar2ego(results, device=pts.device, dtype=pts.dtype)
    pts_new = pts.clone()
    pts_new[:, :3] = transform_xyz_rowvec(pts_new[:, :3], T)
    return points.new_point(pts_new)

def build_ego2lidar(results, device=None, dtype=torch.float32):
    """构建 4x4 变换矩阵: Ego -> LiDAR (传感器)"""
    curr = results["curr"]
    
    # 获取 lidar -> ego 的旋转和平移
    R = Quaternion(curr["lidar2ego_rotation"]).rotation_matrix
    t = np.array(curr["lidar2ego_translation"], dtype=np.float32)
    
    # 计算逆变换 (Rigid Transform Inverse)
    # R_inv = R.T
    # t_inv = -R.T @ t
    R_inv = R.T
    t_inv = -(R_inv @ t)
    
    T_inv = np.eye(4, dtype=np.float32)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    
    T_inv = torch.from_numpy(T_inv)
    if device is not None:
        T_inv = T_inv.to(device)
    return T_inv.to(dtype)

def points_to_lidar(points: BasePoints, results):
    """将 BasePoints 从 Ego 坐标系转换回 LiDAR 传感器坐标系"""
    pts = points.tensor
    T_inv = build_ego2lidar(results, device=pts.device, dtype=pts.dtype)
    
    pts_new = pts.clone()
    # 矩阵乘法形式: P_new = P_old @ R_inv.T + t_inv
    pts_new[:, :3] = pts_new[:, :3].matmul(T_inv[:3, :3].T) + T_inv[:3, 3]
    
    return points.new_point(pts_new)


def make_bev_heightmap_from_xyz_fixbug(
    xyz_ego: torch.Tensor,
    x_bounds=(-40.0, 40.0),
    y_bounds=(-40.0, 40.0),
    z_bounds=(-1.0, 5.4),
    resolution=0.4,
    method="max",          # 'max'|'min'|'mean'
    normalize=True,
    fill_value=255,        # 255 ignore 
):
    """Generate BEV heightmap on ego coordinates.
    Return: (H, W) float tensor
    """
    assert method in ("max", "min", "mean")
    xmin, xmax = x_bounds
    ymin, ymax = y_bounds
    zmin, zmax = z_bounds

    H = int(np.ceil((xmax - xmin) / resolution))
    W = int(np.ceil((ymax - ymin) / resolution))

    if xyz_ego.numel() == 0:
        hm = torch.full((H, W), fill_value, dtype=torch.float32, device=xyz_ego.device)
        return hm.unsqueeze(0)

    x, y, z = xyz_ego[:, 0], xyz_ego[:, 1], xyz_ego[:, 2]
    # xy filter
    m = (x >= xmin) & (x < xmax) & (y >= ymin) & (y < ymax)
    x, y, z = x[m], y[m], z[m]
    if x.numel() == 0:
        hm = torch.full((H, W), fill_value, dtype=torch.float32, device=xyz_ego.device)
        return hm.unsqueeze(0)

    z = torch.clamp(z, min=zmin, max=zmax)

    # ego -> grid indices
    row = torch.floor((x - xmin) / resolution).long()
    col = torch.floor((y - ymin) / resolution).long()

    v = (row >= 0) & (row < H) & (col >= 0) & (col < W)
    row, col, z = row[v], col[v], z[v]
    if row.numel() == 0:
        hm = torch.full((H, W), fill_value, dtype=torch.float32, device=xyz_ego.device)
        raise ValueError("wrong lidar input in pipeline!")

    device = xyz_ego.device
    z = z.float()

    # aggregate
    lin = row * W + col  # (M,)
    if method in ("max", "min") and hasattr(torch.Tensor, "scatter_reduce"):
        flat = torch.full((H * W,), float("-inf") if method == "max" else float("inf"), device=device, dtype=torch.float32)
        reduce = "amax" if method == "max" else "amin"
        flat = flat.scatter_reduce(0, lin, z, reduce=reduce, include_self=True)
        hm = flat.view(H, W)
        hm[hm == (float("-inf") if method == "max" else float("inf"))] = float("nan")
    elif method == "mean":
        flat_sum = torch.zeros((H * W,), device=device, dtype=torch.float32)
        flat_cnt = torch.zeros((H * W,), device=device, dtype=torch.float32)
        flat_sum = flat_sum.scatter_add(0, lin, z)
        flat_cnt = flat_cnt.scatter_add(0, lin, torch.ones_like(z))
        hm = (flat_sum / torch.clamp(flat_cnt, min=1.0)).view(H, W)
        hm[flat_cnt.view(H, W) == 0] = float("nan")
    else:
        # fallback numpy for older torch (max/min)
        row_np = row.detach().cpu().numpy()
        col_np = col.detach().cpu().numpy()
        z_np = z.detach().cpu().numpy()
        if method == "max":
            hm_np = np.full((H, W), -np.inf, dtype=np.float32)
            np.maximum.at(hm_np, (row_np, col_np), z_np)
            hm_np[hm_np == -np.inf] = np.nan
        else:
            hm_np = np.full((H, W), np.inf, dtype=np.float32)
            np.minimum.at(hm_np, (row_np, col_np), z_np)
            hm_np[hm_np == np.inf] = np.nan
        hm = torch.from_numpy(hm_np).to(device)

    height_mask = torch.isfinite(hm)   # 或 ~torch.isnan(hm)
    hm = torch.nan_to_num(hm, nan=fill_value)

    return hm, height_mask


def make_bev_heightmap_from_xyz_top_voxel(
    xyz_ego: torch.Tensor,
    x_bounds=(-40.0, 40.0),
    y_bounds=(-40.0, 40.0),
    z_bounds=(-1.0, 5.4),
    xy_resolution=0.4,
    z_resolution=0.4,
    fill_value=255.0,
    min_points_per_voxel=1,
):
    """
    用 lidar 点先离散到 (x,y,z) voxel，再对每个 (x,y) 取最高 occupied z-bin。
    返回:
        hm: (X, Y) metric height, invalid=fill_value
        height_mask: (X, Y) bool
    """
    device = xyz_ego.device
    dtype = xyz_ego.dtype

    xmin, xmax = x_bounds
    ymin, ymax = y_bounds
    zmin, zmax = z_bounds

    X = int(round((xmax - xmin) / xy_resolution))
    Y = int(round((ymax - ymin) / xy_resolution))
    Z = int(round((zmax - zmin) / z_resolution))

    if xyz_ego.numel() == 0:
        hm = torch.full((X, Y), fill_value, dtype=torch.float32, device=device)
        mask = torch.zeros((X, Y), dtype=torch.bool, device=device)
        return hm, mask

    x = xyz_ego[:, 0]
    y = xyz_ego[:, 1]
    z = xyz_ego[:, 2]

    # 先裁剪到体素范围内
    valid = (
        (x >= xmin) & (x < xmax) &
        (y >= ymin) & (y < ymax) &
        (z >= zmin) & (z < zmax)
    )
    x = x[valid]
    y = y[valid]
    z = z[valid]

    if x.numel() == 0:
        hm = torch.full((X, Y), fill_value, dtype=torch.float32, device=device)
        mask = torch.zeros((X, Y), dtype=torch.bool, device=device)
        return hm, mask

    # metric -> voxel index
    ix = torch.floor((x - xmin) / xy_resolution).long()   # [0, X)
    iy = torch.floor((y - ymin) / xy_resolution).long()   # [0, Y)
    iz = torch.floor((z - zmin) / z_resolution).long()    # [0, Z)

    keep = (
        (ix >= 0) & (ix < X) &
        (iy >= 0) & (iy < Y) &
        (iz >= 0) & (iz < Z)
    )
    ix = ix[keep]
    iy = iy[keep]
    iz = iz[keep]

    if ix.numel() == 0:
        hm = torch.full((X, Y), fill_value, dtype=torch.float32, device=device)
        mask = torch.zeros((X, Y), dtype=torch.bool, device=device)
        return hm, mask

    # 统计每个 (x,y,z) voxel 是否被占据
    lin_xyz = ix * (Y * Z) + iy * Z + iz
    voxel_cnt = torch.zeros((X * Y * Z,), dtype=torch.int32, device=device)
    voxel_cnt.scatter_add_(0, lin_xyz, torch.ones_like(lin_xyz, dtype=torch.int32))

    occupied = voxel_cnt.view(X, Y, Z) >= min_points_per_voxel   # (X, Y, Z)

    # 取 top occupied z index
    z_ids = torch.arange(Z, device=device).view(1, 1, Z)
    z_masked = torch.where(
        occupied,
        z_ids,
        torch.full_like(z_ids, -1)
    )
    top_z = z_masked.max(dim=-1).values   # (X, Y)

    height_mask = top_z >= 0

    # 转 metric，高度取 voxel center
    hm = torch.full((X, Y), fill_value, dtype=torch.float32, device=device)
    hm_valid = zmin + (top_z[height_mask].float() + 1 ) * z_resolution
    hm[height_mask] = hm_valid

    return hm, height_mask




@PIPELINES.register_module(force=True)
class Lidar2EgoAndHeightmap:
    """Pipeline: lidar->ego transform, then generate BEV heightmap in ego."""

    def __init__(
        self,
        surroundocc=False,
        x_bounds=(-50.0, 50.0),
        y_bounds=(-50.0, 50.0),
        resolution=0.2,
        z_bounds=(-5.0, 3.0),
        method="max",
        trans2ego=False,
        normalize_z=True,
        fill_value=0.0,
        save_points_ego=True,     # True: results['points_ego']; False: overwrite results['points']
    ):
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.resolution = resolution
        self.z_bounds = z_bounds
        self.method = method
        self.normalize_z = normalize_z
        self.fill_value = fill_value
        self.save_points_ego = save_points_ego
        self.trans2ego = trans2ego
        self.surroundocc = surroundocc

    def __call__(self, results):
        # 1) lidar -> ego
        points_lidar = results["points"]
                
        if self.trans2ego:
            points = points_to_ego(points_lidar, results)
        else:
            points = points_lidar

        if self.surroundocc:
            points = points_to_lidar(points_lidar, results)


        # 2) heightmap in ego
        xyz_ego = points.tensor[:, :3]
        hm, hm_mask = make_bev_heightmap_from_xyz_fixbug(
            xyz_ego,
            x_bounds=self.x_bounds,
            y_bounds=self.y_bounds,
            resolution=self.resolution,
            z_bounds=self.z_bounds,
            method=self.method,
            normalize=self.normalize_z,
            fill_value=self.fill_value,
        )

        # 3) bev aug !!! 
        if results['rotate_bda'] != 0:
            raise ValueError("rotate bda must be 0 !!!")
        if results.get('flip_dx', False):
            hm =  torch.flip(hm, [0])
            hm_mask = torch.flip(hm_mask, [0])

        if results.get('flip_dy', False):
            hm =  torch.flip(hm, [1])
            hm_mask = torch.flip(hm_mask, [1])
        

        results["bev_height_map"] = hm  
        results["bev_height_mask"] = hm_mask
        # import pdb;pdb.set_trace()
        return results

    def __repr__(self):
        return (f"{self.__class__.__name__}(x_bounds={self.x_bounds}, "
                f"y_bounds={self.y_bounds}, res={self.resolution}, "
                f"z_bounds={self.z_bounds}, method={self.method})")



