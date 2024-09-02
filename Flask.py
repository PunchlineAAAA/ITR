from flask import Flask, request, jsonify
import Retrieval as r

app = Flask(__name__)


# 得到文生图结果
@app.route("searchfor/image", methods=["POST"])
def get_image():
    # 获取POST请求中的JSON数据
    data = request.get_json()

    if not data:
        return jsonify({"error": "No data provided"}), 400

    # 数据处理
    top_k_indices, top_k_similarities = r.get_top_k_similar_images_fast(data, r.image_tensor)

    final_top_k_results = r.get_top_k_similar_image_slow(top_k_indices, r.image_paths, data, top_k_similarities)

    processed_data = {
        "message": final_top_k_results,
        "status": "succeed"
    }

    return jsonify(processed_data), 200


# 得到图生文结果
@app.route("searchfor/text", method=["POST"])
def get_text():
    # 获取POST请求中的JSON数据
    data = request.get_json()

    if not data:
        return jsonify({"error": "No data provided"}), 400

    top_k_indices, top_k_similarities = r.get_top_k_similar_text_fast(data, r.text_tensor, top_k=5)

    final_top_k_results = r.get_top_k_similar_text_slow(top_k_indices, r.text_descs, data, top_k_similarities)

    processed_data = {
        "message": final_top_k_results,
        "status": "succeed"
    }

    return jsonify(processed_data), 200


if __name__ == '__main__':
    app.run(debug=True)