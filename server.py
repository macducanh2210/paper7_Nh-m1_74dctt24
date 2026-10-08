"""Máy chủ web cục bộ, chỉ dùng thư viện chuẩn của Python."""

import json
import os
import sys
import threading
import time
import uuid
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

os.environ.setdefault("MPLBACKEND", "Agg")

import cv2
import matplotlib
import numpy as np

import demo


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
WEB_DIR = ROOT / "web"
STATIC_DIR = WEB_DIR / "static"
RUNS_DIR = ROOT / "output" / "web_runs"
MAX_UPLOAD_BYTES = 32 * 1024 * 1024
MAX_IMAGE_PIXELS = 6_000_000
RESULT_FILES = {
    "01_disparity.png",
    "02_depth.png",
    "03_pointcloud.png",
    "04_bev.png",
    "05_size_vs_depth.png",
    "06_error_vs_depth.png",
    "point_cloud.ply",
}
STATIC_FILES = {"style.css", "app.js"}
PIPELINE_LOCK = threading.Lock()

matplotlib.use("Agg")


def doc_tham_so_calib(noi_dung):
    """Đọc ma trận chiếu camera trái/phải từ nội dung calib."""
    cac_ma_tran = {}
    for dong in noi_dung.splitlines():
        if ":" not in dong:
            continue
        khoa, chuoi_so = dong.split(":", 1)
        khoa = khoa.strip()
        if khoa not in {"P_rect_02", "P_rect_03", "P2", "P3"}:
            continue
        try:
            cac_so = [float(gia_tri) for gia_tri in chuoi_so.split()]
        except ValueError as loi:
            raise ValueError(f"Dòng {khoa} trong calib chứa giá trị không phải số.") from loi
        if len(cac_so) != 12:
            raise ValueError(f"Dòng {khoa} trong calib phải có đúng 12 số.")
        cac_ma_tran[khoa] = np.asarray(cac_so, dtype=np.float64).reshape(3, 4)

    if {"P_rect_02", "P_rect_03"} <= cac_ma_tran.keys():
        trai, phai = cac_ma_tran["P_rect_02"], cac_ma_tran["P_rect_03"]
    elif {"P2", "P3"} <= cac_ma_tran.keys():
        trai, phai = cac_ma_tran["P2"], cac_ma_tran["P3"]
    else:
        raise ValueError(
            "Không tìm thấy đủ P_rect_02/P_rect_03 hoặc P2/P3 trong calib."
        )

    fx = float(trai[0, 0])
    if fx == 0:
        raise ValueError("Giá trị fx trong calibration phải khác 0.")
    tham_so = {
        "fx": fx,
        "fy": float(trai[1, 1]),
        "cx": float(trai[0, 2]),
        "cy": float(trai[1, 2]),
        "baseline": float((trai[0, 3] - phai[0, 3]) / fx),
    }
    if not all(np.isfinite(value) for value in tham_so.values()):
        raise ValueError("Calibration có chứa tham số không hữu hạn.")
    return tham_so


def doc_anh_tai_len(du_lieu, ten):
    """Giải mã ảnh upload và trả lỗi dễ hiểu khi ảnh không hợp lệ."""
    anh = cv2.imdecode(np.frombuffer(du_lieu, dtype=np.uint8), cv2.IMREAD_COLOR)
    if anh is None:
        raise ValueError(f"Không giải mã được ảnh {ten}. Hãy chọn PNG hoặc JPEG hợp lệ.")
    if anh.shape[0] * anh.shape[1] > MAX_IMAGE_PIXELS:
        raise ValueError(f"Ảnh {ten} lớn hơn giới hạn 6 megapixel để bảo vệ bộ nhớ CPU.")
    return anh


def doc_anh_mau(ten):
    """Đọc ảnh mẫu trong data/ theo đường dẫn cố định của ứng dụng."""
    duong_dan = DATA_DIR / ten
    if not duong_dan.is_file():
        raise ValueError(f"Không tìm thấy ảnh mẫu {duong_dan}.")
    anh = cv2.imread(str(duong_dan), cv2.IMREAD_COLOR)
    if anh is None:
        raise ValueError(f"Không đọc được ảnh mẫu {duong_dan}.")
    if anh.shape[0] * anh.shape[1] > MAX_IMAGE_PIXELS:
        raise ValueError(f"Ảnh mẫu {ten} lớn hơn giới hạn 6 megapixel.")
    return anh


def lay_cau_hinh():
    """Trả trạng thái bộ ảnh mẫu và calibration mẫu."""
    du_lieu = {"samples_available": True, "sample_calibration": None}
    try:
        du_lieu["sample_calibration"] = doc_tham_so_calib(
            (DATA_DIR / "calib.txt").read_text(encoding="utf-8-sig")
        )
    except (OSError, ValueError) as loi:
        du_lieu["samples_available"] = all(
            (DATA_DIR / ten).is_file() for ten in ("left.png", "right.png")
        )
        du_lieu["calibration_warning"] = str(loi)

    try:
        du_lieu["sample_intrinsics_hint"] = demo.doc_calib(
            DATA_DIR / "calib.txt"
        )[:4]
    except (OSError, ValueError):
        du_lieu["sample_intrinsics_hint"] = [721.5377, 721.5377, 609.5593, 172.854]
    return du_lieu


def tao_ket_qua(form, files):
    """Chạy đầy đủ disparity, depth, point cloud và các biểu đồ."""
    bat_dau_tong = time.perf_counter()
    dung_anh_mau = form.get("use_sample") == "true"

    if dung_anh_mau:
        anh_trai, anh_phai = doc_anh_mau("left.png"), doc_anh_mau("right.png")
    else:
        if "left" not in files or "right" not in files:
            raise ValueError("Hãy tải lên đủ ảnh trái và ảnh phải.")
        anh_trai = doc_anh_tai_len(files["left"]["data"], "trái")
        anh_phai = doc_anh_tai_len(files["right"]["data"], "phải")

    if anh_trai.shape[:2] != anh_phai.shape[:2]:
        raise ValueError(
            "Kích thước hai ảnh không bằng nhau: "
            f"{anh_trai.shape[1]}×{anh_trai.shape[0]} và "
            f"{anh_phai.shape[1]}×{anh_phai.shape[0]}."
        )

    noi_dung_calib = ""
    if "calib" in files:
        try:
            noi_dung_calib = files["calib"]["data"].decode("utf-8-sig")
        except UnicodeDecodeError as loi:
            raise ValueError("File calibration không phải văn bản UTF-8 hợp lệ.") from loi
    if not noi_dung_calib:
        noi_dung_calib = form.get("calib_text", "")
    if dung_anh_mau and not noi_dung_calib:
        duong_dan_calib = DATA_DIR / "calib.txt"
        if duong_dan_calib.is_file():
            noi_dung_calib = duong_dan_calib.read_text(encoding="utf-8-sig")

    tham_so = {}
    canh_bao = None
    if noi_dung_calib.strip():
        try:
            tham_so = doc_tham_so_calib(noi_dung_calib)
        except ValueError as loi:
            canh_bao = f"Không dùng được calibration đã tải: {loi}"

    for ten in ("fx", "fy", "cx", "cy", "baseline"):
        gia_tri = form.get(ten, "").strip()
        if gia_tri:
            try:
                tham_so[ten] = float(gia_tri)
            except ValueError as loi:
                raise ValueError(f"Thông số {ten} phải là số.") from loi

    thieu = [ten for ten in ("fx", "fy", "cx", "cy", "baseline") if ten not in tham_so]
    if thieu:
        raise ValueError(
            "Thiếu tham số camera: "
            + ", ".join(thieu)
            + ". Hãy tải calibration đúng hoặc nhập đủ các ô tham số."
        )
    if not all(np.isfinite(value) for value in tham_so.values()):
        raise ValueError("Thông số camera phải là số hữu hạn.")
    if any(tham_so[ten] <= 0 for ten in ("fx", "fy", "baseline")):
        raise ValueError("fx, fy và baseline phải lớn hơn 0.")

    try:
        ti_le = float(form.get("scale", "1"))
        gioi_han_depth = float(form.get("max_depth", "80"))
    except ValueError as loi:
        raise ValueError("Tỉ lệ ảnh và giới hạn độ sâu phải là số.") from loi
    if not np.isfinite(ti_le) or not 0.15 <= ti_le <= 1.0:
        raise ValueError("Tỉ lệ ảnh phải nằm trong khoảng 0,15 đến 1,00.")
    if not np.isfinite(gioi_han_depth) or not 1 <= gioi_han_depth <= 200:
        raise ValueError("Giới hạn độ sâu phải nằm trong khoảng 1 đến 200 m.")

    kich_thuoc_goc = [int(anh_trai.shape[1]), int(anh_trai.shape[0])]
    if ti_le != 1.0:
        kich_thuoc = (
            max(1, round(anh_trai.shape[1] * ti_le)),
            max(1, round(anh_trai.shape[0] * ti_le)),
        )
        anh_trai = cv2.resize(anh_trai, kich_thuoc, interpolation=cv2.INTER_AREA)
        anh_phai = cv2.resize(anh_phai, kich_thuoc, interpolation=cv2.INTER_AREA)
        for ten in ("fx", "fy", "cx", "cy"):
            tham_so[ten] *= ti_le

    if anh_trai.shape[1] <= 128:
        raise ValueError("Ảnh sau resize phải rộng hơn 128 px.")

    run_id = uuid.uuid4().hex
    thu_muc = RUNS_DIR / run_id
    thu_muc.mkdir(parents=True, exist_ok=False)
    thoi_gian = {}

    def do_buoc(ten, ham):
        bat_dau = time.perf_counter()
        ket_qua = ham()
        thoi_gian[ten] = round(time.perf_counter() - bat_dau, 3)
        return ket_qua

    disparity = do_buoc("Disparity", lambda: demo.tinh_disparity(anh_trai, anh_phai))
    hop_le = disparity > 0
    ti_le_hop_le = 100.0 * np.count_nonzero(hop_le) / disparity.size
    gia_tri_disparity = disparity[hop_le]
    vmax_disparity = (
        max(1.0, float(np.percentile(gia_tri_disparity, 99)))
        if gia_tri_disparity.size
        else 1.0
    )
    plt = demo.cau_hinh_matplotlib(True)
    do_buoc(
        "Lưu disparity",
        lambda: demo.luu_anh_mau(
            plt,
            thu_muc / "01_disparity.png",
            np.where(hop_le, disparity, np.nan),
            "Bản đồ disparity từ ảnh stereo",
            "plasma",
            "Disparity (pixel)",
            vmax_disparity,
        ),
    )

    depth = do_buoc(
        "Depth", lambda: demo.tinh_depth(disparity, tham_so["fx"], tham_so["baseline"])
    )
    gia_tri_depth = depth[depth > 0]
    do_buoc(
        "Lưu depth",
        lambda: demo.luu_anh_mau(
            plt,
            thu_muc / "02_depth.png",
            np.where(depth > 0, np.minimum(depth, gioi_han_depth), np.nan),
            f"Bản đồ độ sâu (hiển thị tối đa {gioi_han_depth:g} m)",
            "viridis",
            "Độ sâu (m)",
            gioi_han_depth,
        ),
    )

    diem, mau = do_buoc(
        "Point cloud",
        lambda: demo.tao_point_cloud(
            depth,
            anh_trai,
            tham_so["fx"],
            tham_so["fy"],
            tham_so["cx"],
            tham_so["cy"],
        ),
    )
    do_buoc(
        "Lưu point cloud",
        lambda: demo.luu_ply(thu_muc / "point_cloud.ply", diem, mau),
    )
    fig = do_buoc(
        "Vẽ point cloud",
        lambda: demo.ve_point_cloud(plt, thu_muc / "03_pointcloud.png", diem, mau),
    )
    plt.close(fig)
    do_buoc("BEV", lambda: demo.ve_bev(plt, thu_muc / "04_bev.png", diem))
    do_buoc(
        "Kích thước xe",
        lambda: demo.ve_kich_thuoc_xe(
            plt, thu_muc / "05_size_vs_depth.png", tham_so["fx"]
        ),
    )
    do_buoc(
        "Sai số depth",
        lambda: demo.ve_sai_so_depth(
            plt,
            thu_muc / "06_error_vs_depth.png",
            tham_so["fx"],
            tham_so["baseline"],
        ),
    )

    so_diem_hien_thi = min(len(diem), 18000)
    if len(diem) > so_diem_hien_thi:
        chi_so = np.random.default_rng(0).choice(
            len(diem), so_diem_hien_thi, replace=False
        )
        diem_hien_thi, mau_hien_thi = diem[chi_so], mau[chi_so]
    else:
        diem_hien_thi, mau_hien_thi = diem, mau

    return {
        "run_id": run_id,
        "warning": canh_bao,
        "metrics": {
            "width": int(anh_trai.shape[1]),
            "height": int(anh_trai.shape[0]),
            "original_width": kich_thuoc_goc[0],
            "original_height": kich_thuoc_goc[1],
            "fx": round(tham_so["fx"], 3),
            "fy": round(tham_so["fy"], 3),
            "cx": round(tham_so["cx"], 3),
            "cy": round(tham_so["cy"], 3),
            "baseline": round(tham_so["baseline"], 5),
            "valid_percent": round(ti_le_hop_le, 2),
            "point_count": len(diem),
            "depth_min": round(float(gia_tri_depth.min()), 2)
            if gia_tri_depth.size
            else None,
            "depth_max": round(float(gia_tri_depth.max()), 2)
            if gia_tri_depth.size
            else None,
            "depth_median": round(float(np.median(gia_tri_depth)), 2)
            if gia_tri_depth.size
            else None,
            "elapsed": round(time.perf_counter() - bat_dau_tong, 2),
            "steps": thoi_gian,
        },
        "points": np.column_stack((diem_hien_thi, mau_hien_thi)).round(3).tolist(),
        "files": {ten: f"/result/{run_id}/{ten}" for ten in sorted(RESULT_FILES)},
    }


def tao_cau_hinh():
    """Cung cấp thông tin cho trang, không làm lộ đường dẫn nội bộ."""
    du_lieu = {"samples_available": True, "sample_calibration": None}
    try:
        du_lieu["sample_calibration"] = doc_tham_so_calib(
            (DATA_DIR / "calib.txt").read_text(encoding="utf-8-sig")
        )
    except (OSError, ValueError) as loi:
        du_lieu["samples_available"] = all(
            (DATA_DIR / ten).is_file() for ten in ("left.png", "right.png")
        )
        du_lieu["calibration_warning"] = str(loi)
    try:
        du_lieu["sample_intrinsics_hint"] = demo.doc_calib(
            DATA_DIR / "calib.txt"
        )[:4]
    except (OSError, ValueError):
        du_lieu["sample_intrinsics_hint"] = [721.5377, 721.5377, 609.5593, 172.854]
    return du_lieu


class WebHandler(BaseHTTPRequestHandler):
    """Phục vụ trang tĩnh và API xử lý ảnh trên localhost."""

    server_version = "PseudoLidarDemo/1.0"

    def send_bytes(self, du_lieu, content_type, status=200, attachment=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(du_lieu)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        if attachment:
            self.send_header(
                "Content-Disposition", f'attachment; filename="{attachment}"'
            )
        self.end_headers()
        self.wfile.write(du_lieu)

    def send_json(self, data, status=200):
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_bytes(payload, "application/json; charset=utf-8", status)

    def do_GET(self):
        parsed = urlsplit(self.path)
        path = unquote(parsed.path)
        if path == "/":
            return self.send_bytes(
                (WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8"
            )
        if path == "/api/config":
            return self.send_json(tao_cau_hinh())
        if path.startswith("/static/"):
            filename = path.removeprefix("/static/")
            if filename not in STATIC_FILES:
                return self.send_error(404)
            content_type = (
                "text/css; charset=utf-8"
                if filename.endswith(".css")
                else "text/javascript; charset=utf-8"
            )
            return self.send_bytes((STATIC_DIR / filename).read_bytes(), content_type)
        if path.startswith("/result/"):
            parts = path.strip("/").split("/")
            if len(parts) != 3:
                return self.send_error(404)
            _, run_id, filename = parts
            if (
                len(run_id) != 32
                or any(ch not in "0123456789abcdef" for ch in run_id)
                or filename not in RESULT_FILES
            ):
                return self.send_error(404)
            duong_dan = RUNS_DIR / run_id / filename
            if not duong_dan.is_file():
                return self.send_error(404)
            content_type = (
                "image/png" if filename.endswith(".png") else "application/octet-stream"
            )
            attachment = filename if filename.endswith(".ply") else None
            return self.send_bytes(duong_dan.read_bytes(), content_type, attachment=attachment)
        return self.send_error(404)

    def do_POST(self):
        if urlsplit(self.path).path != "/api/run":
            return self.send_error(404)
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self.send_json({"error": "Header Content-Length không hợp lệ."}, 400)
        if length <= 0:
            return self.send_json({"error": "Không nhận được dữ liệu đầu vào."}, 400)
        if length > MAX_UPLOAD_BYTES:
            return self.send_json(
                {"error": "Tổng kích thước tệp tải lên vượt quá 32 MB."}, 413
            )
        if not self.headers.get("Content-Type", "").lower().startswith(
            "multipart/form-data"
        ):
            return self.send_json(
                {"error": "Yêu cầu cần có dữ liệu form multipart/form-data."}, 415
            )

        body = self.rfile.read(length)
        content_type = self.headers["Content-Type"]
        message = BytesParser(policy=policy.default).parsebytes(
            b"MIME-Version: 1.0\r\nContent-Type: "
            + content_type.encode("ascii")
            + b"\r\n\r\n"
            + body
        )
        form = {}
        files = {}
        for part in message.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if not name:
                continue
            payload = part.get_payload(decode=True) or b""
            filename = part.get_filename()
            if filename is not None:
                files[name] = {"filename": filename, "data": payload}
            else:
                try:
                    form[name] = payload.decode(part.get_content_charset() or "utf-8")
                except UnicodeDecodeError:
                    return self.send_json(
                        {"error": f"Trường {name} không phải văn bản UTF-8 hợp lệ."}, 400
                    )

        try:
            with PIPELINE_LOCK:
                ket_qua = tao_ket_qua(form, files)
            return self.send_json(ket_qua)
        except ValueError as loi:
            return self.send_json({"error": str(loi)}, 400)
        except cv2.error as loi:
            return self.send_json({"error": f"OpenCV không xử lý được ảnh: {loi}"}, 400)
        except OSError as loi:
            print(f"Lỗi đọc/ghi kết quả: {loi}", file=sys.stderr)
            return self.send_json({"error": f"Lỗi đọc hoặc lưu tệp: {loi}"}, 500)
        except Exception as loi:
            print(f"Lỗi không dự kiến trong pipeline: {loi}", file=sys.stderr)
            return self.send_json(
                {"error": "Có lỗi nội bộ khi xử lý. Kiểm tra console máy chủ để biết thêm."},
                500,
            )

    def log_message(self, format_string, *args):
        sys.stderr.write(
            f"{self.log_date_time_string()} {format_string % args}\n"
        )


def main():
    """Khởi động web server chỉ lắng nghe trên máy cục bộ."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 5000), WebHandler)
    print("Mở trình duyệt tại http://127.0.0.1:5000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng máy chủ demo.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
