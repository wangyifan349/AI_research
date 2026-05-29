"""
依赖库安装命令：

    pip install flask face_recognition pillow numpy

如果安装 face_recognition / dlib 失败，通常需要先安装编译环境。

    sudo apt update
    sudo apt install -y cmake build-essential python3-dev

本程序是一个基于 Flask 的高精度人脸相似度对比系统。用户可以在网页中上传两张人脸图片，
前端使用 jQuery Ajax 将图片提交到后端 API；后端使用 face_recognition 的 CNN 检测模型
查找图片中的主脸，并使用 large 特征点模型与多次采样提取稳定的人脸特征，最终返回人脸
欧氏距离、相似度展示分数、是否匹配、检测到的人脸数量以及模型参数。程序还提供 /help
页面，用于说明接口调用方式、请求参数、返回字段和调用示例。
"""

from flask import Flask, request, jsonify, render_template_string
from PIL import Image, UnidentifiedImageError
import face_recognition
import numpy as np
import io
import time
import traceback


app = Flask(__name__)  # 创建 Flask 应用对象


app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 限制单次请求最大 10MB，避免大文件拖垮服务

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}  # 允许上传的图片格式

FACE_TOLERANCE = 0.6  # 人脸匹配阈值，距离小于等于该值时判断为可能同一人

MAX_IMAGE_SIZE = 1600  # 图片最大边长，精度优先配置会保留较大图像

UPSAMPLE_TIMES = 2  # CNN 检测上采样次数，数值越大越容易检测小脸，但速度越慢

NUM_JITTERS = 10  # 人脸编码采样次数，数值越大结果越稳定，但速度越慢

LANDMARK_MODEL = "large"  # 使用 large 特征点模型，精度高于 small

DETECTION_MODEL = "cnn"  # 使用 CNN 人脸检测模型，精度高于 HOG

try:
    RESAMPLE_FILTER = Image.Resampling.LANCZOS  # Pillow 新版本的高质量缩放滤镜
except AttributeError:
    RESAMPLE_FILTER = Image.LANCZOS  # Pillow 旧版本兼容写法


# =========================================================
# 第一部分：人脸对比核心函数
# =========================================================

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS  # 判断扩展名是否合法


def load_image_from_upload(file_storage) -> np.ndarray:
    filename = file_storage.filename or ""  # 获取上传文件名

    if not allowed_file(filename):  # 校验图片格式
        raise ValueError("仅支持 jpg、jpeg、png、webp 格式图片")

    try:
        image_bytes = file_storage.read()  # 读取上传文件二进制内容
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # 打开图片并转成 RGB
    except UnidentifiedImageError:
        raise ValueError("图片文件无法识别，请上传有效图片")

    image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE), RESAMPLE_FILTER)  # 按比例缩放到最大边长以内

    return np.array(image)  # 转为 numpy 数组，供 face_recognition 使用


def get_largest_face_box(image_array: np.ndarray):
    face_locations = face_recognition.face_locations(  # 检测图片中的所有人脸
        image_array,
        number_of_times_to_upsample=UPSAMPLE_TIMES,
        model=DETECTION_MODEL
    )

    if not face_locations:  # 没有检测到人脸
        return None, 0

    def face_area(box):
        top, right, bottom, left = box  # face_recognition 的人脸框格式
        return max(0, right - left) * max(0, bottom - top)  # 计算人脸框面积

    largest_face_box = max(face_locations, key=face_area)  # 多人脸时选择面积最大的人脸

    return largest_face_box, len(face_locations)  # 返回主脸框和检测到的人脸数量


def extract_face_encoding(image_array: np.ndarray, face_box):
    encodings = face_recognition.face_encodings(  # 提取 128 维人脸特征向量
        image_array,
        known_face_locations=[face_box],
        num_jitters=NUM_JITTERS,
        model=LANDMARK_MODEL
    )

    if not encodings:  # 特征提取失败
        return None

    return encodings[0]  # 返回主脸特征向量


def distance_to_similarity(distance: float, threshold: float = FACE_TOLERANCE) -> float:
    if distance <= threshold:  # 距离小于阈值时，映射到 50 到 100 分
        score = 50 + (threshold - distance) / threshold * 50
    else:  # 距离大于阈值时，映射到 0 到 50 分
        score = 50 - (distance - threshold) / (1.0 - threshold) * 50

    return round(max(0, min(100, score)), 2)  # 限制分数范围为 0 到 100


def compare_faces_core(file1, file2) -> dict:
    start_time = time.time()  # 记录接口处理开始时间

    image1 = load_image_from_upload(file1)  # 读取第一张图片
    image2 = load_image_from_upload(file2)  # 读取第二张图片

    face_box_1, face_count_1 = get_largest_face_box(image1)  # 检测第一张图片主脸
    face_box_2, face_count_2 = get_largest_face_box(image2)  # 检测第二张图片主脸

    if face_box_1 is None:  # 第一张图片没有检测到人脸
        raise ValueError("第一张图片未检测到人脸，请上传清晰、正脸、光线充足的图片")

    if face_box_2 is None:  # 第二张图片没有检测到人脸
        raise ValueError("第二张图片未检测到人脸，请上传清晰、正脸、光线充足的图片")

    encoding1 = extract_face_encoding(image1, face_box_1)  # 提取第一张图片人脸特征
    encoding2 = extract_face_encoding(image2, face_box_2)  # 提取第二张图片人脸特征

    if encoding1 is None:  # 第一张图片特征提取失败
        raise ValueError("第一张图片的人脸特征提取失败")

    if encoding2 is None:  # 第二张图片特征提取失败
        raise ValueError("第二张图片的人脸特征提取失败")

    distance = face_recognition.face_distance([encoding1], encoding2)[0]  # 计算两个人脸特征的欧氏距离
    distance = float(distance)  # 转成普通 float，方便 JSON 序列化

    matched = distance <= FACE_TOLERANCE  # 根据阈值判断是否可能为同一人
    similarity = distance_to_similarity(distance, FACE_TOLERANCE)  # 生成页面展示用相似度
    elapsed_ms = round((time.time() - start_time) * 1000, 2)  # 计算处理耗时，单位毫秒

    return {
        "matched": bool(matched),  # 是否匹配
        "distance": round(distance, 4),  # 欧氏距离，越小越相似
        "similarity": similarity,  # 展示用相似度，不是严格概率
        "threshold": FACE_TOLERANCE,  # 当前匹配阈值
        "face_count_1": face_count_1,  # 第一张图片检测到的人脸数量
        "face_count_2": face_count_2,  # 第二张图片检测到的人脸数量
        "elapsed_ms": elapsed_ms,  # 本次后端处理耗时
        "model": {
            "detection": DETECTION_MODEL,  # 人脸检测模型
            "landmark": LANDMARK_MODEL,  # 人脸特征点模型
            "upsample_times": UPSAMPLE_TIMES,  # 检测上采样次数
            "num_jitters": NUM_JITTERS,  # 编码采样次数
            "max_image_size": MAX_IMAGE_SIZE  # 图片最大边长
        }
    }


# =========================================================
# 第二部分：Flask 页面和 API 路由
# =========================================================

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)  # 返回首页页面


@app.route("/help", methods=["GET"])
def help_page():
    return render_template_string(HELP_HTML)  # 返回 API 帮助页面


@app.route("/api/face/compare", methods=["POST"])
def api_face_compare():
    if "image1" not in request.files or "image2" not in request.files:  # 校验两个上传字段是否存在
        return jsonify({
            "ok": False,
            "error": "请同时上传 image1 和 image2 两张图片"
        }), 400

    try:
        result = compare_faces_core(  # 调用核心人脸对比函数
            request.files["image1"],
            request.files["image2"]
        )

        return jsonify({
            "ok": True,
            "data": result
        })

    except ValueError as e:
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 400

    except Exception:
        app.logger.error(traceback.format_exc())  # 服务端记录完整异常，前端只返回简洁错误
        return jsonify({
            "ok": False,
            "error": "服务器处理失败，请稍后重试"
        }), 500


@app.errorhandler(413)
def file_too_large(_):
    return jsonify({
        "ok": False,
        "error": "图片过大，请上传 10MB 以内的图片"
    }), 413


# =========================================================
# 第三部分：首页前端页面：Bootstrap + jQuery Ajax
# 页面自定义主题色均使用 rgb / rgba，第三个蓝色通道值均为 0。
# =========================================================

INDEX_HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>人脸相似度对比系统</title>

  <link
    href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
    rel="stylesheet"
  >

  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>

  <style>
    :root {
      --theme-main: rgb(255, 80, 0);
      --theme-dark: rgb(200, 40, 0);
      --theme-deep: rgb(90, 20, 0);
      --theme-light: rgb(255, 235, 0);
      --theme-soft: rgb(255, 180, 0);
      --theme-panel: rgb(255, 248, 0);
      --theme-page: rgb(255, 242, 0);
      --theme-muted: rgb(120, 80, 0);
      --theme-text: rgb(45, 25, 0);
      --theme-black: rgb(0, 0, 0);
    }

    * {
      box-shadow-color: rgba(0, 0, 0, 0);
    }

    body {
      min-height: 100vh;
      color: var(--theme-text);
      background:
        radial-gradient(circle at top left, rgba(255, 120, 0, 0.40) 0, rgba(255, 120, 0, 0) 34%),
        linear-gradient(135deg, rgb(255, 245, 0) 0%, rgb(255, 250, 0) 48%, rgb(255, 228, 0) 100%);
    }

    a {
      color: var(--theme-dark);
    }

    a:hover {
      color: var(--theme-deep);
    }

    .hero {
      padding: 58px 0 30px;
      text-align: center;
    }

    .brand-badge {
      display: inline-flex;
      align-items: center;
      background: var(--theme-light);
      color: var(--theme-dark);
      border: 1px solid var(--theme-soft);
      border-radius: 999px;
      padding: 8px 15px;
      font-size: 14px;
      font-weight: 700;
    }

    .hero-title {
      margin-top: 18px;
      font-weight: 900;
      letter-spacing: -0.04em;
      color: var(--theme-deep);
    }

    .hero-desc {
      max-width: 780px;
      margin: 14px auto 0;
      color: var(--theme-muted);
      line-height: 1.8;
    }

    .main-card {
      border: 1px solid var(--theme-soft);
      border-radius: 30px;
      background: var(--theme-panel);
      box-shadow: 0 24px 70px rgba(160, 40, 0, 0.20);
      overflow: hidden;
    }

    .main-card-header {
      background: linear-gradient(135deg, rgb(255, 90, 0), rgb(190, 30, 0));
      color: rgb(255, 245, 0);
      padding: 24px 28px;
    }

    .upload-box {
      height: 100%;
      border: 2px dashed rgb(255, 150, 0);
      border-radius: 24px;
      padding: 22px;
      background: rgb(255, 250, 0);
      transition: all 0.2s ease;
    }

    .upload-box:hover {
      border-color: var(--theme-main);
      background: rgb(255, 238, 0);
    }

    .upload-box.has-image {
      border-style: solid;
      border-color: var(--theme-main);
      background: rgb(255, 232, 0);
    }

    .form-label {
      color: var(--theme-deep);
    }

    .form-control {
      color: var(--theme-text);
      background-color: rgb(255, 245, 0);
      border-color: rgb(220, 100, 0);
    }

    .form-control:focus {
      color: var(--theme-text);
      background-color: rgb(255, 248, 0);
      border-color: var(--theme-main);
      box-shadow: 0 0 0 0.25rem rgba(255, 80, 0, 0.25);
    }

    .preview-img {
      display: none;
      width: 100%;
      height: 300px;
      object-fit: cover;
      margin-top: 16px;
      border-radius: 18px;
      background: rgb(80, 60, 0);
      border: 1px solid rgb(210, 90, 0);
    }

    .btn-theme {
      --bs-btn-color: rgb(255, 245, 0);
      --bs-btn-bg: var(--theme-main);
      --bs-btn-border-color: var(--theme-main);
      --bs-btn-hover-color: rgb(255, 245, 0);
      --bs-btn-hover-bg: var(--theme-dark);
      --bs-btn-hover-border-color: var(--theme-dark);
      --bs-btn-focus-shadow-rgb: 255, 80, 0;
      padding: 12px 34px;
      border-radius: 999px;
      font-weight: 800;
    }

    .btn-outline-theme {
      color: var(--theme-dark);
      border-color: var(--theme-main);
      background: rgb(255, 245, 0);
      border-radius: 999px;
      font-weight: 700;
    }

    .btn-outline-theme:hover {
      color: rgb(255, 245, 0);
      background: var(--theme-main);
      border-color: var(--theme-main);
    }

    .result-card {
      display: none;
      border: 1px solid var(--theme-soft);
      border-radius: 26px;
      background: rgb(255, 250, 0);
      padding: 28px;
    }

    .score-circle {
      width: 142px;
      height: 142px;
      border-radius: 50%;
      margin: auto;
      display: grid;
      place-items: center;
      background: conic-gradient(var(--theme-main) 0deg, rgb(235, 210, 0) 0deg);
    }

    .score-inner {
      width: 112px;
      height: 112px;
      border-radius: 50%;
      background: rgb(255, 245, 0);
      display: grid;
      place-items: center;
      color: var(--theme-dark);
      font-size: 25px;
      font-weight: 900;
    }

    .metric {
      height: 100%;
      padding: 17px;
      border-radius: 18px;
      background: rgb(255, 238, 0);
      border: 1px solid rgb(255, 160, 0);
    }

    .metric-label {
      color: var(--theme-muted);
      font-size: 13px;
    }

    .metric-value {
      margin-top: 4px;
      color: var(--theme-deep);
      font-size: 22px;
      font-weight: 900;
    }

    .note {
      color: var(--theme-muted);
      font-size: 13px;
      line-height: 1.8;
    }

    .alert {
      border-radius: 18px;
    }

    .alert-danger {
      color: rgb(90, 0, 0);
      background-color: rgb(255, 210, 0);
      border-color: rgb(255, 80, 0);
    }

    .text-success {
      color: rgb(40, 120, 0) !important;
    }

    .text-danger {
      color: rgb(180, 0, 0) !important;
    }

    .footer-note {
      margin-top: 22px;
      color: var(--theme-muted);
      font-size: 13px;
      text-align: center;
      line-height: 1.8;
    }

    .spinner-border {
      color: rgb(255, 245, 0);
    }
  </style>
</head>

<body>
  <main class="container">
    <section class="hero">
      <div class="brand-badge">CNN 高精度人脸检测</div>
      <h1 class="display-5 hero-title">人脸相似度对比系统</h1>
      <p class="hero-desc">
        上传两张包含人脸的图片，后端使用 CNN 检测人脸，并使用 large 模型与多次采样提取稳定特征。
        多人脸图片会自动选择面积最大的人脸作为对比对象。
      </p>
      <div class="mt-3">
        <a class="btn btn-outline-theme btn-sm" href="/help">查看 API 帮助</a>
      </div>
    </section>

    <section class="row justify-content-center pb-5">
      <div class="col-12 col-xl-10">
        <div class="card main-card">
          <div class="main-card-header">
            <div class="d-flex flex-column flex-md-row justify-content-between gap-2">
              <div>
                <h5 class="mb-1 fw-bold">上传图片</h5>
                <div class="opacity-75">支持 JPG、PNG、WEBP，单张图片最大 10MB。</div>
              </div>
              <div class="opacity-75 align-self-md-center small">
                Ajax 接口：POST /api/face/compare
              </div>
            </div>
          </div>

          <div class="card-body p-4 p-md-5">
            <form id="compareForm" enctype="multipart/form-data">
              <div class="row g-4">
                <div class="col-md-6">
                  <div class="upload-box" id="box1">
                    <label class="form-label fw-bold" for="image1">第一张图片</label>
                    <input class="form-control" type="file" id="image1" name="image1" accept="image/*" required>
                    <img id="preview1" class="preview-img" alt="第一张图片预览">
                  </div>
                </div>

                <div class="col-md-6">
                  <div class="upload-box" id="box2">
                    <label class="form-label fw-bold" for="image2">第二张图片</label>
                    <input class="form-control" type="file" id="image2" name="image2" accept="image/*" required>
                    <img id="preview2" class="preview-img" alt="第二张图片预览">
                  </div>
                </div>
              </div>

              <div id="errorBox" class="alert alert-danger mt-4 d-none"></div>

              <div class="d-grid d-md-flex justify-content-md-center mt-4">
                <button id="submitBtn" type="submit" class="btn btn-theme btn-lg">开始对比</button>
              </div>
            </form>

            <div id="resultCard" class="result-card mt-5">
              <div class="row g-4 align-items-center">
                <div class="col-md-4 text-center">
                  <div class="score-circle" id="scoreCircle">
                    <div class="score-inner">
                      <span id="similarityText">--%</span>
                    </div>
                  </div>
                  <div id="matchText" class="mt-3 fw-bold"></div>
                </div>

                <div class="col-md-8">
                  <div class="row g-3">
                    <div class="col-sm-4">
                      <div class="metric">
                        <div class="metric-label">欧氏距离</div>
                        <div class="metric-value" id="distanceText">--</div>
                      </div>
                    </div>

                    <div class="col-sm-4">
                      <div class="metric">
                        <div class="metric-label">匹配阈值</div>
                        <div class="metric-value" id="thresholdText">--</div>
                      </div>
                    </div>

                    <div class="col-sm-4">
                      <div class="metric">
                        <div class="metric-label">检测人脸数</div>
                        <div class="metric-value" id="faceCountText">--</div>
                      </div>
                    </div>
                  </div>

                  <div class="row g-3 mt-1">
                    <div class="col-sm-3">
                      <div class="metric">
                        <div class="metric-label">检测模型</div>
                        <div class="metric-value fs-5" id="detectModelText">--</div>
                      </div>
                    </div>

                    <div class="col-sm-3">
                      <div class="metric">
                        <div class="metric-label">采样次数</div>
                        <div class="metric-value fs-5" id="jitterText">--</div>
                      </div>
                    </div>

                    <div class="col-sm-3">
                      <div class="metric">
                        <div class="metric-label">上采样</div>
                        <div class="metric-value fs-5" id="upsampleText">--</div>
                      </div>
                    </div>

                    <div class="col-sm-3">
                      <div class="metric">
                        <div class="metric-label">耗时</div>
                        <div class="metric-value fs-5" id="elapsedText">--</div>
                      </div>
                    </div>
                  </div>

                  <p class="note mt-3 mb-0">
                    判断依据：欧氏距离越小，人脸越相似。系统主要依据“欧氏距离 ≤ 阈值”判断是否可能为同一人。
                    页面上的相似度是展示分数，不等同于严格概率。
                  </p>
                </div>
              </div>
            </div>

          </div>
        </div>

        <div class="footer-note">
          当前配置为精度优先：CNN 检测、large 特征点模型、上采样 2 次、编码采样 10 次。
          没有 GPU/CUDA 时会比较慢，这是高精度配置的正常现象。
        </div>
      </div>
    </section>
  </main>

  <script>
    function bindPreview(inputSelector, previewSelector, boxSelector) {
      $(inputSelector).on("change", function () {
        const file = this.files[0]; // 获取用户选择的图片文件

        if (!file) {
          return; // 没有选择文件时直接结束
        }

        const previewUrl = URL.createObjectURL(file); // 生成本地预览地址

        $(previewSelector)
          .attr("src", previewUrl)
          .fadeIn(150); // 显示图片预览

        $(boxSelector).addClass("has-image"); // 标记上传框已经有图片
      });
    }

    function showError(message) {
      $("#errorBox")
        .text(message)
        .removeClass("d-none"); // 显示错误提示

      $("#resultCard").hide(); // 出错时隐藏结果区
    }

    function clearError() {
      $("#errorBox")
        .text("")
        .addClass("d-none"); // 清空并隐藏错误提示
    }

    function setLoading(loading) {
      const $btn = $("#submitBtn"); // 获取提交按钮

      if (loading) {
        $btn.prop("disabled", true); // 请求中禁用按钮
        $btn.html('<span class="spinner-border spinner-border-sm me-2"></span>正在高精度对比...');
      } else {
        $btn.prop("disabled", false); // 请求结束恢复按钮
        $btn.text("开始对比");
      }
    }

    function renderResult(data) {
      const similarity = Number(data.similarity); // 读取相似度分数
      const degrees = Math.max(0, Math.min(100, similarity)) * 3.6; // 转换为圆环角度

      $("#similarityText").text(similarity.toFixed(2) + "%"); // 渲染相似度
      $("#distanceText").text(data.distance); // 渲染欧氏距离
      $("#thresholdText").text(data.threshold); // 渲染阈值
      $("#faceCountText").text(data.face_count_1 + " / " + data.face_count_2); // 渲染人脸数量
      $("#detectModelText").text(data.model.detection); // 渲染检测模型
      $("#jitterText").text(data.model.num_jitters); // 渲染采样次数
      $("#upsampleText").text(data.model.upsample_times); // 渲染上采样次数
      $("#elapsedText").text(data.elapsed_ms + "ms"); // 渲染后端耗时

      if (data.matched) {
        $("#matchText")
          .text("判断结果：可能是同一人")
          .removeClass("text-danger")
          .addClass("text-success"); // 匹配时显示绿色倾向文字
      } else {
        $("#matchText")
          .text("判断结果：不像同一人")
          .removeClass("text-success")
          .addClass("text-danger"); // 不匹配时显示红色倾向文字
      }

      $("#scoreCircle").css(
        "background",
        `conic-gradient(var(--theme-main) ${degrees}deg, rgb(235, 210, 0) 0deg)`
      ); // 更新相似度圆环

      $("#resultCard").fadeIn(180); // 显示结果卡片

      document.getElementById("resultCard").scrollIntoView({
        behavior: "smooth",
        block: "center"
      }); // 滚动到结果区域
    }

    bindPreview("#image1", "#preview1", "#box1"); // 绑定第一张图片预览
    bindPreview("#image2", "#preview2", "#box2"); // 绑定第二张图片预览

    $("#compareForm").on("submit", function (event) {
      event.preventDefault(); // 阻止表单默认提交

      clearError(); // 清空旧错误

      const file1 = $("#image1")[0].files[0]; // 获取第一张图片
      const file2 = $("#image2")[0].files[0]; // 获取第二张图片

      if (!file1 || !file2) {
        showError("请先选择两张图片"); // 前端基础校验
        return;
      }

      const formData = new FormData(); // 创建 Ajax 上传数据
      formData.append("image1", file1); // 添加第一张图片
      formData.append("image2", file2); // 添加第二张图片

      setLoading(true); // 进入加载状态

      $.ajax({
        url: "/api/face/compare",
        type: "POST",
        data: formData,
        processData: false,
        contentType: false,
        dataType: "json",

        success: function (res) {
          if (!res.ok) {
            showError(res.error || "对比失败，请稍后重试"); // 显示业务错误
            return;
          }

          renderResult(res.data); // 渲染成功结果
        },

        error: function (xhr) {
          let message = "请求失败，请检查服务是否正常运行"; // 默认错误文案

          if (xhr.responseJSON && xhr.responseJSON.error) {
            message = xhr.responseJSON.error; // 优先使用后端返回错误
          }

          showError(message); // 显示错误
        },

        complete: function () {
          setLoading(false); // 请求完成后恢复按钮
        }
      });
    });
  </script>
</body>
</html>
"""


# =========================================================
# 第四部分：帮助页面
# 页面自定义主题色均使用 rgb / rgba，第三个蓝色通道值均为 0。
# =========================================================

HELP_HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>API 帮助 - 人脸相似度对比系统</title>

  <link
    href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
    rel="stylesheet"
  >

  <style>
    :root {
      --theme-main: rgb(255, 80, 0);
      --theme-dark: rgb(200, 40, 0);
      --theme-deep: rgb(90, 20, 0);
      --theme-light: rgb(255, 235, 0);
      --theme-soft: rgb(255, 180, 0);
      --theme-panel: rgb(255, 248, 0);
      --theme-page: rgb(255, 242, 0);
      --theme-muted: rgb(120, 80, 0);
      --theme-text: rgb(45, 25, 0);
      --theme-black: rgb(0, 0, 0);
    }

    body {
      min-height: 100vh;
      color: var(--theme-text);
      background:
        radial-gradient(circle at top left, rgba(255, 120, 0, 0.35) 0, rgba(255, 120, 0, 0) 34%),
        linear-gradient(135deg, rgb(255, 245, 0) 0%, rgb(255, 250, 0) 48%, rgb(255, 228, 0) 100%);
    }

    a {
      color: var(--theme-dark);
    }

    a:hover {
      color: var(--theme-deep);
    }

    .page-wrap {
      max-width: 960px;
      margin: 0 auto;
      padding: 48px 16px;
    }

    .help-card {
      border: 1px solid var(--theme-soft);
      border-radius: 26px;
      background: var(--theme-panel);
      box-shadow: 0 20px 60px rgba(160, 40, 0, 0.18);
      overflow: hidden;
    }

    .help-header {
      background: linear-gradient(135deg, rgb(255, 90, 0), rgb(190, 30, 0));
      color: rgb(255, 245, 0);
      padding: 26px 30px;
    }

    .badge-soft {
      display: inline-block;
      background: var(--theme-light);
      color: var(--theme-dark);
      border: 1px solid var(--theme-soft);
      padding: 6px 12px;
      border-radius: 999px;
      font-weight: 700;
      font-size: 13px;
    }

    pre {
      background: rgb(35, 20, 0);
      color: rgb(255, 245, 0);
      padding: 18px;
      border-radius: 16px;
      overflow-x: auto;
      border: 1px solid rgb(180, 80, 0);
    }

    code {
      color: var(--theme-dark);
    }

    pre code {
      color: rgb(255, 245, 0);
    }

    .btn-theme {
      --bs-btn-color: rgb(255, 245, 0);
      --bs-btn-bg: var(--theme-main);
      --bs-btn-border-color: var(--theme-main);
      --bs-btn-hover-color: rgb(255, 245, 0);
      --bs-btn-hover-bg: var(--theme-dark);
      --bs-btn-hover-border-color: var(--theme-dark);
      --bs-btn-focus-shadow-rgb: 255, 80, 0;
      border-radius: 999px;
      font-weight: 700;
    }

    .table {
      --bs-table-color: var(--theme-text);
      --bs-table-bg: rgb(255, 248, 0);
      --bs-table-border-color: rgb(230, 120, 0);
      color: var(--theme-text);
      border-color: rgb(230, 120, 0);
    }

    .table th {
      white-space: nowrap;
      color: var(--theme-deep);
      background: rgb(255, 230, 0);
    }

    .table td {
      background: rgb(255, 248, 0);
    }

    .alert-warning {
      color: rgb(90, 40, 0);
      background-color: rgb(255, 220, 0);
      border-color: rgb(255, 140, 0);
    }
  </style>
</head>

<body>
  <main class="page-wrap">
    <div class="mb-3">
      <a href="/" class="btn btn-theme btn-sm">返回首页</a>
    </div>

    <div class="card help-card">
      <div class="help-header">
        <div class="badge-soft mb-3">API Help</div>
        <h1 class="h3 fw-bold mb-2">人脸相似度对比接口说明</h1>
        <p class="mb-0 opacity-75">
          本页面说明网站前端和第三方程序如何调用后端人脸对比接口。
        </p>
      </div>

      <div class="card-body p-4 p-md-5">
        <h2 class="h5 fw-bold">1. 页面地址</h2>
        <table class="table table-bordered align-middle mt-3">
          <tbody>
            <tr>
              <th>首页</th>
              <td><code>GET /</code></td>
            </tr>
            <tr>
              <th>帮助页</th>
              <td><code>GET /help</code></td>
            </tr>
          </tbody>
        </table>

        <h2 class="h5 fw-bold mt-5">2. 人脸对比 API</h2>
        <table class="table table-bordered align-middle mt-3">
          <tbody>
            <tr>
              <th>接口地址</th>
              <td><code>POST /api/face/compare</code></td>
            </tr>
            <tr>
              <th>请求类型</th>
              <td><code>multipart/form-data</code></td>
            </tr>
            <tr>
              <th>参数 image1</th>
              <td>第一张图片文件，支持 jpg、jpeg、png、webp</td>
            </tr>
            <tr>
              <th>参数 image2</th>
              <td>第二张图片文件，支持 jpg、jpeg、png、webp</td>
            </tr>
            <tr>
              <th>单次请求大小</th>
              <td>最大 10MB</td>
            </tr>
          </tbody>
        </table>

        <h2 class="h5 fw-bold mt-5">3. curl 调用示例</h2>
        <pre><code>curl -X POST http://127.0.0.1:5000/api/face/compare \\
  -F "image1=@a.jpg" \\
  -F "image2=@b.jpg"</code></pre>

        <h2 class="h5 fw-bold mt-5">4. jQuery Ajax 调用示例</h2>
        <pre><code>const formData = new FormData();
formData.append("image1", file1);
formData.append("image2", file2);

$.ajax({
  url: "/api/face/compare",
  type: "POST",
  data: formData,
  processData: false,
  contentType: false,
  dataType: "json",
  success: function (res) {
    console.log(res);
  }
});</code></pre>

        <h2 class="h5 fw-bold mt-5">5. 成功返回示例</h2>
        <pre><code>{
  "ok": true,
  "data": {
    "matched": true,
    "distance": 0.3821,
    "similarity": 68.16,
    "threshold": 0.6,
    "face_count_1": 1,
    "face_count_2": 1,
    "elapsed_ms": 2380.15,
    "model": {
      "detection": "cnn",
      "landmark": "large",
      "upsample_times": 2,
      "num_jitters": 10,
      "max_image_size": 1600
    }
  }
}</code></pre>

        <h2 class="h5 fw-bold mt-5">6. 失败返回示例</h2>
        <pre><code>{
  "ok": false,
  "error": "第一张图片未检测到人脸，请上传清晰、正脸、光线充足的图片"
}</code></pre>

        <h2 class="h5 fw-bold mt-5">7. 字段说明</h2>
        <table class="table table-bordered align-middle mt-3">
          <thead>
            <tr>
              <th>字段</th>
              <th>说明</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><code>matched</code></td>
              <td>是否判断为可能同一人，依据是 <code>distance <= threshold</code></td>
            </tr>
            <tr>
              <td><code>distance</code></td>
              <td>人脸特征欧氏距离，越小越相似</td>
            </tr>
            <tr>
              <td><code>similarity</code></td>
              <td>页面展示用相似度分数，不是严格概率</td>
            </tr>
            <tr>
              <td><code>threshold</code></td>
              <td>匹配阈值，当前为 0.6</td>
            </tr>
            <tr>
              <td><code>face_count_1</code></td>
              <td>第一张图片检测到的人脸数量</td>
            </tr>
            <tr>
              <td><code>face_count_2</code></td>
              <td>第二张图片检测到的人脸数量</td>
            </tr>
            <tr>
              <td><code>elapsed_ms</code></td>
              <td>后端处理耗时，单位为毫秒</td>
            </tr>
            <tr>
              <td><code>model</code></td>
              <td>当前使用的人脸检测模型、特征点模型、采样次数等配置</td>
            </tr>
          </tbody>
        </table>

        <div class="alert alert-warning mt-4 mb-0">
          当前配置是精度优先：CNN 检测、large 特征点模型、上采样 2 次、编码采样 10 次。
          如果服务器没有 GPU/CUDA，接口响应会比较慢。
        </div>
      </div>
    </div>
  </main>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)  # 本地开发启动方式，生产环境建议使用 gunicorn 并关闭 debug
