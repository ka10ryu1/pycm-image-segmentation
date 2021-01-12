# pycm-image-segmantation

Deep learning（セグメンテーション）に PyCM を使う場合のサンプルです

コードや PyCM に関する解説は以下のブログを参照してください（記事公開後にリンクを更新）。  
[Deep learning 等の精度評価に便利な PyCM の紹介と各種指標の比較](https://tech-blog.optim.co.jp/)

コードの大部分は [pytorch/example/mnist](https://github.com/pytorch/examples/tree/master/mnist) に準拠しています。

## 使い方

### ライブラリのインストール

以下のコマンドを入力してください

```bash
pip install pycm opencv-python
```

PyTorch は環境によって違うと思うので、[公式](https://pytorch.org/get-started/locally/) に従ってください。

### バージョン

以下のバージョンで動作確認済みです

- Python: 3.8.6
- PyTroch: 1.7.0
- torchvision: 0.8.1
- PyCM: 3.0
- OpenCV: 4.4.0.46

### 実行

これで動きます。

```bash
python ./main.py
```

各種オプション引数があります。

```bash
python ./main.py -h
```

例えば、一度学習で作成した重みを再利用して推論だけしたい場合は以下のようにします。

```bash
python ./main.py --weight ./out/unet_weight.pt
```

学習用データを確認する場合は、以下のようにします。

```bash
python ./generate.py
```
