from flask import Flask, request, jsonify
import Retrieval as r

# 初始化 app
app = Flask(__name__)


# 得到文生图结果
@app.route("/searchfor/image", methods=["POST"])
def get_image():
    # 获取POST请求中的JSON数据
    data = request.get_json()

    # 合法判断
    if not data:
        return jsonify({
            "code": 400,
            "error": "No data provided",
        })

    data = data["keywords"]

    # 文生图
    top_k_indices, top_k_similarities = r.get_top_k_similar_images_fast(data, r.image_tensor)

    final_top_k_results = r.get_top_k_similar_image_slow(top_k_indices, r.image_paths, data, top_k_similarities)

    # 统一格式
    prefix = "./dataset/images/"
    final_top_k_results = [i[len(prefix):] for i in final_top_k_results]

    # 返还信息
    processed_data = {
        "code": 200,
        "message": final_top_k_results,
    }

    return jsonify(processed_data)


# 得到图生文结果
@app.route("/searchfor/text", methods=["POST"])
def get_text():
    # 获取POST请求中的JSON数据
    data = request.get_json()

    # 合法判断
    if not data:
        return jsonify({
            "code": 400,
            "error": "No data provided"
        })

    data = data["keywords"]

    # 图升文
    top_k_indices, top_k_similarities = r.get_top_k_similar_text_fast(data, r.text_tensor, top_k=5)

    final_top_k_results = r.get_top_k_similar_text_slow(top_k_indices, r.text_descs, data, top_k_similarities)

    # 返还信息
    processed_data = {
        "code": 200,
        "message": final_top_k_results,
    }

    return jsonify(processed_data)


if __name__ == '__main__':
    # 约定传入内容格式：
    # {
    #   keywords: 内容
    # }
    app.run(debug=True, host="0.0.0.0", port=8000)