"""Minh họa tạo pseudo-LiDAR từ ảnh stereo bằng StereoSGBM trên CPU."""

import argparse
import sys
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np


def tinh_thoi_gian(ten_buoc, ham):
    """Chạy một bước, in thời gian và trả lại kết quả."""
    bat_dau = time.perf_counter()
    ket_qua = ham()
    print(f"Thời gian {ten_buoc}: {time.perf_counter() - bat_dau:.2f} giây")
    return ket_qua


def doc_calib(duong_dan):
    """Đọc ma trận chiếu camera trái/phải từ calib.txt."""
    cac_ma_tran = {}
    cac_khoa = {"P_rect_02", "P_rect_03", "P2", "P3"}

    with duong_dan.open("r", encoding="utf-8") as tep:
        for dong in tep:
            if ":" not in dong:
                continue
            khoa, chuoi_so = dong.split(":", 1)
            khoa = khoa.strip()
            if khoa not in cac_khoa:
                continue
            try:
                gia_tri = [float(so) for so in chuoi_so.split()]
            except ValueError as loi:
                raise ValueError(f"Dòng {khoa} trong calib không phải là số hợp lệ.") from loi
            if len(gia_tri) != 12:
                raise ValueError(f"Dòng {khoa} cần đúng 12 số, hiện có {len(gia_tri)}.")
            cac_ma_tran[khoa] = np.asarray(gia_tri, dtype=np.float64).reshape(3, 4)

    if "P_rect_02" in cac_ma_tran and "P_rect_03" in cac_ma_tran:
        trai, phai = cac_ma_tran["P_rect_02"], cac_ma_tran["P_rect_03"]
    elif "P2" in cac_ma_tran and "P3" in cac_ma_tran:
        trai, phai = cac_ma_tran["P2"], cac_ma_tran["P3"]
    else:
        raise ValueError(
            "Không tìm thấy đủ P_rect_02/P_rect_03 hoặc P2/P3 trong file calib."
        )

    fx = float(trai[0, 0])
    fy = float(trai[1, 1])
    cx = float(trai[0, 2])
    cy = float(trai[1, 2])
    baseline = float((trai[0, 3] - phai[0, 3]) / fx)
    return fx, fy, cx, cy, baseline


def doc_du_lieu(duong_dan_trai, duong_dan_phai, ti_le):
    """Kiểm tra, đọc và resize cặp ảnh đầu vào."""
    anh_trai = cv2.imread(str(duong_dan_trai), cv2.IMREAD_COLOR)
    anh_phai = cv2.imread(str(duong_dan_phai), cv2.IMREAD_COLOR)
    if anh_trai is None:
        raise ValueError(f"Không đọc được ảnh trái: {duong_dan_trai}")
    if anh_phai is None:
        raise ValueError(f"Không đọc được ảnh phải: {duong_dan_phai}")
    if anh_trai.shape[:2] != anh_phai.shape[:2]:
        raise ValueError(
            "Kích thước ảnh trái và ảnh phải không bằng nhau: "
            f"{anh_trai.shape[1]}x{anh_trai.shape[0]} và "
            f"{anh_phai.shape[1]}x{anh_phai.shape[0]}."
        )

    if ti_le != 1.0:
        kich_thuoc = (
            max(1, round(anh_trai.shape[1] * ti_le)),
            max(1, round(anh_trai.shape[0] * ti_le)),
        )
        anh_trai = cv2.resize(anh_trai, kich_thuoc, interpolation=cv2.INTER_AREA)
        anh_phai = cv2.resize(anh_phai, kich_thuoc, interpolation=cv2.INTER_AREA)
    return anh_trai, anh_phai


def tinh_disparity(anh_trai, anh_phai):
    """Tính disparity stereo bằng StereoSGBM chạy trên CPU."""
    rong = anh_trai.shape[1]
    if rong <= 128:
        raise ValueError(
            f"Ảnh sau resize rộng {rong}px; cần rộng hơn 128px cho numDisparities=128. "
            "Hãy tăng --scale."
        )

    xam_trai = cv2.cvtColor(anh_trai, cv2.COLOR_BGR2GRAY)
    xam_phai = cv2.cvtColor(anh_phai, cv2.COLOR_BGR2GRAY)
    block_size = 5
    bo_tinh = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,
        blockSize=block_size,
        P1=8 * block_size**2,
        P2=32 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return bo_tinh.compute(xam_trai, xam_phai).astype(np.float32) / 16.0


def tinh_depth(disparity, fx, baseline):
    """Đổi disparity dương sang độ sâu z (mét)."""
    depth = np.zeros(disparity.shape, dtype=np.float32)
    hop_le = disparity > 0
    depth[hop_le] = (fx * baseline) / disparity[hop_le]
    return depth


def tao_point_cloud(depth, anh_trai, fx, fy, cx, cy):
    """Chiếu các pixel có depth hợp lệ thành điểm 3D và lấy màu ảnh."""
    hop_le = (depth > 0) & (depth < 80.0)
    v, u = np.nonzero(hop_le)
    z = depth[v, u]
    x = (u.astype(np.float32) - cx) * z / fx
    y = (v.astype(np.float32) - cy) * z / fy
    diem = np.column_stack((x, y, z)).astype(np.float32)
    mau = anh_trai[v, u, ::-1].astype(np.float32) / 255.0
    return diem, mau


def luu_ply(duong_dan, diem, mau):
    """Lưu point cloud dạng PLY nhị phân, không phụ thuộc Open3D."""
    kieu_diem = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    du_lieu = np.empty(len(diem), dtype=kieu_diem)
    if len(diem):
        du_lieu["x"], du_lieu["y"], du_lieu["z"] = diem.T
        rgb = np.clip(mau * 255.0, 0, 255).astype(np.uint8)
        du_lieu["red"], du_lieu["green"], du_lieu["blue"] = rgb.T

    tieu_de = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(diem)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with duong_dan.open("wb") as tep:
        tep.write(tieu_de.encode("ascii"))
        du_lieu.tofile(tep)


def cau_hinh_matplotlib(khong_mo_cua_so):
    """Chọn backend và font hiển thị tiếng Việt."""
    if khong_mo_cua_so:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def luu_anh_mau(plt, duong_dan, du_lieu, tieu_de, ten_bang_mau, nhan_mau, vmax):
    """Lưu ảnh disparity hoặc depth kèm colorbar."""
    fig, ax = plt.subplots(figsize=(12, 4))
    anh_hien_thi = np.ma.masked_invalid(du_lieu)
    im = ax.imshow(
        anh_hien_thi,
        cmap=ten_bang_mau,
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(tieu_de)
    ax.set_xlabel("Tọa độ ngang (pixel)")
    ax.set_ylabel("Tọa độ dọc (pixel)")
    fig.colorbar(im, ax=ax, label=nhan_mau)
    fig.tight_layout()
    fig.savefig(duong_dan, dpi=150, bbox_inches="tight")
    plt.close(fig)


def ve_point_cloud(plt, duong_dan, diem, mau):
    """Vẽ point cloud từ hai góc nhìn, lấy mẫu tối đa 50.000 điểm."""
    so_mau = min(len(diem), 50_000)
    if len(diem) > so_mau:
        chi_so = np.random.default_rng(0).choice(len(diem), so_mau, replace=False)
        diem, mau = diem[chi_so], mau[chi_so]

    fig = plt.figure(figsize=(13, 5))
    tieu_de = ("Nhìn phía trước", "Nhìn nghiêng")
    goc_nhin = ((0, -90), (20, -60))
    for vi_tri, (ten, (goc_cao, goc_quay)) in enumerate(zip(tieu_de, goc_nhin), 1):
        ax = fig.add_subplot(1, 2, vi_tri, projection="3d")
        if len(diem):
            ax.scatter(diem[:, 0], diem[:, 1], diem[:, 2], c=mau, s=0.25, linewidths=0)
        else:
            ax.text2D(0.25, 0.5, "Không có điểm hợp lệ", transform=ax.transAxes)
        ax.set_title(ten)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.view_init(elev=goc_cao, azim=goc_quay)
    fig.suptitle("Đám mây điểm pseudo-LiDAR")
    fig.tight_layout()
    fig.savefig(duong_dan, dpi=150, bbox_inches="tight")
    return fig


def mo_cua_so_open3d(diem, mau):
    """Hiển thị point cloud với Open3D; trả False để dùng matplotlib nếu lỗi."""
    try:
        import open3d as o3d
    except Exception as loi:
        print(f"Không tải được Open3D ({loi}); sẽ dùng cửa sổ matplotlib 3D.")
        return False

    try:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(diem.astype(np.float64))
        cloud.colors = o3d.utility.Vector3dVector(mau.astype(np.float64))
        bien_doi = np.eye(4)
        bien_doi[1, 1] = -1
        cloud.transform(bien_doi)
        o3d.visualization.draw_geometries(
            [cloud], window_name="Pseudo-LiDAR - Open3D"
        )
        return True
    except Exception as loi:
        print(f"Không mở được cửa sổ Open3D ({loi}); sẽ dùng cửa sổ matplotlib 3D.")
        return False


def ve_bev(plt, duong_dan, diem):
    """Tạo BEV dạng histogram, giới hạn độ cao để giảm điểm mặt đất."""
    chon = (diem[:, 1] > -1.0) & (diem[:, 1] < 1.4)
    x, z = diem[chon, 0], diem[chon, 2]
    bev, _, _ = np.histogram2d(
        x,
        z,
        bins=(400, 600),
        range=((-20, 20), (0, 60)),
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        np.log1p(bev.T),
        origin="lower",
        cmap="gray",
        extent=(-20, 20, 0, 60),
        aspect="auto",
    )
    ax.set_title("Góc nhìn từ trên xuống (BEV)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    fig.colorbar(im, ax=ax, label="Mật độ điểm (log(1 + số điểm))")
    fig.tight_layout()
    fig.savefig(duong_dan, dpi=150, bbox_inches="tight")
    plt.close(fig)


def ve_kich_thuoc_xe(plt, duong_dan, fx):
    """So sánh bề rộng xe tính theo pixel với bề rộng thật."""
    z = np.linspace(5, 80, 300)
    pixel_width = fx * 1.8 / z
    fig, ax_pixel = plt.subplots(figsize=(8, 5))
    duong_pixel = ax_pixel.plot(z, pixel_width, color="tab:blue", label="Bề rộng trên ảnh")
    ax_pixel.set_xlabel("Khoảng cách z (m)")
    ax_pixel.set_ylabel("Bề rộng trên ảnh (pixel)", color="tab:blue")
    ax_pixel.tick_params(axis="y", labelcolor="tab:blue")
    ax_pixel.grid(True, alpha=0.3)

    ax_that = ax_pixel.twinx()
    duong_that = ax_that.plot(
        z, np.full_like(z, 1.8), "--", color="tab:orange", label="Bề rộng thật"
    )
    ax_that.set_ylabel("Bề rộng thật (m)", color="tab:orange")
    ax_that.tick_params(axis="y", labelcolor="tab:orange")
    ax_pixel.set_title("Kích thước xe thay đổi theo khoảng cách")
    ax_pixel.legend(duong_pixel + duong_that, ["Bề rộng trên ảnh", "Bề rộng thật"])
    fig.tight_layout()
    fig.savefig(duong_dan, dpi=150, bbox_inches="tight")
    plt.close(fig)


def ve_sai_so_depth(plt, duong_dan, fx, baseline):
    """Vẽ sai số độ sâu ứng với một số mức sai số disparity."""
    z = np.linspace(5, 80, 300)
    fig, ax = plt.subplots(figsize=(8, 5))
    for delta_d in (0.25, 0.5, 1.0):
        sai_so = z**2 / (fx * baseline) * delta_d
        ax.plot(z, sai_so, label=f"Sai số disparity = {delta_d:g} pixel")
    ax.set_title("Sai số độ sâu theo khoảng cách")
    ax.set_xlabel("Khoảng cách thật z (m)")
    ax.set_ylabel("Sai số độ sâu (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(duong_dan, dpi=150, bbox_inches="tight")
    plt.close(fig)


def tao_bo_tham_so():
    """Khai báo các tham số dòng lệnh."""
    bo_phan_tich = argparse.ArgumentParser(
        description="Minh họa pseudo-LiDAR từ cặp ảnh KITTI stereo."
    )
    bo_phan_tich.add_argument("--left", type=Path, default=Path("data/left.png"))
    bo_phan_tich.add_argument("--right", type=Path, default=Path("data/right.png"))
    bo_phan_tich.add_argument("--calib", type=Path, default=Path("data/calib.txt"))
    bo_phan_tich.add_argument("--output", type=Path, default=Path("output"))
    bo_phan_tich.add_argument("--scale", type=float, default=1.0)
    bo_phan_tich.add_argument("--fx", type=float)
    bo_phan_tich.add_argument("--fy", type=float)
    bo_phan_tich.add_argument("--cx", type=float)
    bo_phan_tich.add_argument("--cy", type=float)
    bo_phan_tich.add_argument("--baseline", type=float)
    bo_phan_tich.add_argument(
        "--no-window", action="store_true", help="Chỉ lưu ảnh, không mở cửa sổ."
    )
    return bo_phan_tich.parse_args()


def main():
    """Chạy tuần tự các bước minh họa và lưu kết quả."""
    args = tao_bo_tham_so()
    if not np.isfinite(args.scale) or args.scale <= 0:
        print("Lỗi: --scale phải là số dương.", file=sys.stderr)
        return 1
    if not args.calib.is_file():
        print(f"Lỗi: Không tìm thấy file calib: {args.calib}", file=sys.stderr)
        return 1

    try:
        anh_trai, anh_phai = tinh_thoi_gian(
            "đọc và resize ảnh",
            lambda: doc_du_lieu(args.left, args.right, args.scale),
        )
        fx, fy, cx, cy, baseline = tinh_thoi_gian(
            "đọc tham số camera", lambda: doc_calib(args.calib)
        )
    except (OSError, ValueError) as loi:
        print(f"Lỗi dữ liệu đầu vào: {loi}", file=sys.stderr)
        return 1

    fx = args.fx if args.fx is not None else fx
    fy = args.fy if args.fy is not None else fy
    cx = args.cx if args.cx is not None else cx
    cy = args.cy if args.cy is not None else cy
    baseline = args.baseline if args.baseline is not None else baseline
    if not all(np.isfinite(v) and v > 0 for v in (fx, fy, baseline)):
        print("Lỗi: fx, fy và baseline phải là các số dương hữu hạn.", file=sys.stderr)
        return 1
    if not all(np.isfinite(v) for v in (cx, cy)):
        print("Lỗi: cx và cy phải là các số hữu hạn.", file=sys.stderr)
        return 1

    if args.scale != 1.0:
        fx, fy, cx, cy = (gia_tri * args.scale for gia_tri in (fx, fy, cx, cy))
    print(f"Kích thước ảnh sau resize: {anh_trai.shape[1]}x{anh_trai.shape[0]} pixel")
    print(
        f"fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}, "
        f"baseline={baseline:.6f} m"
    )
    args.output.mkdir(parents=True, exist_ok=True)

    try:
        plt = cau_hinh_matplotlib(args.no_window)
        disparity = tinh_thoi_gian(
            "tính disparity", lambda: tinh_disparity(anh_trai, anh_phai)
        )
        hop_le = disparity > 0
        ti_le_hop_le = 100.0 * np.count_nonzero(hop_le) / disparity.size
        print(f"Pixel có disparity hợp lệ: {ti_le_hop_le:.2f}%")
        gia_tri_disp = disparity[hop_le]
        vmax_disp = max(1.0, float(np.percentile(gia_tri_disp, 99))) if gia_tri_disp.size else 1.0
        tinh_thoi_gian(
            "lưu ảnh disparity",
            lambda: luu_anh_mau(
                plt,
                args.output / "01_disparity.png",
                np.where(hop_le, disparity, np.nan),
                "Bản đồ disparity từ ảnh stereo",
                "plasma",
                "Disparity (pixel)",
                vmax_disp,
            ),
        )

        depth = tinh_thoi_gian("tính depth", lambda: tinh_depth(disparity, fx, baseline))
        gia_tri_depth = depth[depth > 0]
        if gia_tri_depth.size:
            print(
                "Depth hợp lệ: nhỏ nhất "
                f"{gia_tri_depth.min():.2f} m, lớn nhất {gia_tri_depth.max():.2f} m, "
                f"trung vị {np.median(gia_tri_depth):.2f} m"
            )
        else:
            print("Depth hợp lệ: không có pixel nào.")
        tinh_thoi_gian(
            "lưu ảnh depth",
            lambda: luu_anh_mau(
                plt,
                args.output / "02_depth.png",
                np.where(depth > 0, np.minimum(depth, 80.0), np.nan),
                "Bản đồ độ sâu (giới hạn hiển thị 80 m)",
                "viridis",
                "Độ sâu (m)",
                80.0,
            ),
        )

        diem, mau = tinh_thoi_gian(
            "tạo point cloud",
            lambda: tao_point_cloud(depth, anh_trai, fx, fy, cx, cy),
        )
        print(f"Số điểm trong point cloud: {len(diem)}")
        tinh_thoi_gian(
            "lưu point cloud PLY",
            lambda: luu_ply(args.output / "point_cloud.ply", diem, mau),
        )
        fig_cloud = tinh_thoi_gian(
            "vẽ point cloud matplotlib",
            lambda: ve_point_cloud(
                plt, args.output / "03_pointcloud.png", diem, mau
            ),
        )

        da_mo_open3d = False
        if not args.no_window:
            da_mo_open3d = tinh_thoi_gian(
                "hiển thị Open3D", lambda: mo_cua_so_open3d(diem, mau)
            )
        if da_mo_open3d:
            plt.close(fig_cloud)

        tinh_thoi_gian(
            "tạo ảnh BEV",
            lambda: ve_bev(plt, args.output / "04_bev.png", diem),
        )
        tinh_thoi_gian(
            "vẽ kích thước xe",
            lambda: ve_kich_thuoc_xe(
                plt, args.output / "05_size_vs_depth.png", fx
            ),
        )
        tinh_thoi_gian(
            "vẽ sai số độ sâu",
            lambda: ve_sai_so_depth(
                plt, args.output / "06_error_vs_depth.png", fx, baseline
            ),
        )
        if not args.no_window and not da_mo_open3d:
            plt.show()
        print(f"Đã lưu kết quả trong: {args.output.resolve()}")
        return 0
    except (OSError, ValueError, cv2.error) as loi:
        print(f"Lỗi khi xử lý dữ liệu: {loi}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
