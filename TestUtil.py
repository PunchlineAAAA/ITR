from model.ImageTextCrossAttention import ImageTextCrossAttention
from model.ImageFeatureExtractor import  ImageFeatureExtractor
from model.TextFeatureExtractor import TextFeatureExtractor
from model.Loss import Loss


# text: 文本描述（可以多个）
# image: 图像（可以多个）
# return: 相似度矩阵
def test_attention(text, image):
    image_to_text_attention_model = ImageTextCrossAttention()
    return image_to_text_attention_model(text, image)


# images: 图像（可以多个）
# return: 图像特征
def test_image_feature_extractor(images):
    model = ImageFeatureExtractor()
    features = model(images)
    return features


# texts: 文本（可以多个）
# return: 文本特征
def test_text_feature_extractor(texts):
    model = TextFeatureExtractor()
    features = model(texts)
    return features


def test_loss(texts, images):
    model = Loss()
    loss = model(texts, images)
    return loss



if __name__ == "__main__":
    # image_paths = ["./dataset/images/667626_18933d713e.jpg", "./dataset/images/3637013_c675de7705.jpg"]
    # text = ["A girl", "Water"]
    # print(test_attention(text, image_paths))

    # image_paths = ["./dataset/images/667626_18933d713e.jpg", "./dataset/images/3637013_c675de7705.jpg"]
    # image_feature = test_image_feature_extractor(image_paths)
    # print(image_feature)

    # text = ["A girl", "Water"]
    # text_feature = test_text_feature_extractor(text)
    # print(text_feature)

    image_paths = ["./dataset/images/667626_18933d713e.jpg", "./dataset/images/3637013_c675de7705.jpg"]
    text = ["A girl", "Water"]
    loss = test_loss(text, image_paths)
    print(loss)