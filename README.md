# 跨模态图文检索系统

&emsp;该仓库是东南大学暑期实训工程项目：跨模态图文检索系统的模型实现。该仓库中存储的代码是项目最终的模型实现（不包含模型训练参数）。

## 部署

&emsp;为了部署这个系统，你需要进行以下操作：

&emsp;首先，你需要使用如下代码克隆这个 git 仓库

    git clone https://github.com/GodTheHands/ITR.git

将仓库克隆到本地文件夹后，需要使用以下代码下载其中 `requirements.txt` 文件夹声明的依赖包

    pip install -r requirements.txt

在完成上述操作后，因为一些设置，你还需要进行如下操作：

- 首先，因为 `torch` 安装包过于庞大，在国内网络条件下不易安装，所以这里并没有在 `requirements.txt` 文件中声明，你可以使用代理或其他方式下载它，例如 `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
- 本项目使用 Python 3.9 解释器，为了避免依赖版本冲突，应当尽量配置 Python 3.9 解释器
- 你应当在目录文件夹下新建 `saved_models` 文件夹以存放模型参数文件以及记录文件
- 在 `model` 文件夹下的 `TextFeatureExtractor.py` 中，`model_name` 定义了加载 `BERT` 模型的路径，项目默认是下载到本地，并在 `model` 文件夹下新建 `bert` 文件夹来存储模型的 `config.json`、`pytorch_model.bin` 和 `vocab.txt` 文件。如果可以顺利连接到 HuggingFace 上的模型，则直接参考 `Transformers` 库的 `from_pretrained` 函数使用示例
- 在 `dataset` 文件夹下，你应当下载 `flickr8k` 数据集，并将其中的 `images` 文件夹放在该文件夹下

## 训练

&emsp;如果你希望训练这个模型，请直接运行 `Train.py` 文件，等待其训练输出。

## 检索

&emsp;这个系统的检索是基于 `open_clip` 包的，为了加速检索过程，代码中将数据库内容进行嵌入并存储。为了完成这个缓存过程，你应该直接运行 `FastCaching.py`，最终结果会输出到 `.\dataset\output.pt` 中。

&emsp;而后，如果你想执行检索，请打开 `Retrieval.py`，按照主函数中给出的案例（包括注释）修改代码为自己的检索目标，并执行。

## API

&emsp;如果你想要用这个模型构建一个 API，并使用 HTTP 请求传输数据给外界。请运行 `Flask.py` 文件。并且，在执行图片搜索文本时，请要保证图片在本地对应路径存在，如果外部传输图片不存在于数据集中，请在外部编写接口将图片下载到本地。

## 评估

&emsp;如果你想要评估这个模型，请在 `RKCalculator.py` 中调用对应函数，结果为函数名对应过程的 R@K1, R@K5 和 R@K10。