# Employee Inventions: Evolutionary Equilibrium and Institutional Shocks (Replication Package)

本リポジトリは、進化ゲーム理論に基づき、日本の職務発明制度における二つの主要な制度的ショック（2004年の中村事件判決と2015年特許法改正）がR&Dインセンティブに与えた異質な影響を分析した論文の再現パッケージです。

この研究は、法制度の変更に対する企業の反応が、ショックの性質（「リスク」対「明確性」）によって大きく異なるという、従来のインセンティブ理論を超越した**「リスク対明確性パラドックス」**を実証しています。

## 論文情報

* **論文タイトル**: Evolving Ownership and Employee Inventions: Equilibrium and Shocks Revealed by Japanese Patent Data
* **著者**: Atsushi Tsuneki and Koki Arai
* **連絡先**: 荒井 弘毅 (Koki Arai) / koki.arai@nifty.ne.jp
* **主要な発見**: 
    1.  どちらのショックも、発明者インセンティブと特許の質（$\ln v, \ln q$）を**一貫して抑制**した（効率性損失）。
    2.  企業活動量（$\ln w$）については、**リスクショック（2004年）で促進**（防御的拡張）、**明確性シフト（2015年）で抑制**（合理化・選別）という、**「リスク対明確性パラドックス」**を発見した。

## ファイル構成

| ファイル名 | 説明 | 貢献セクション |
| :--- | :--- | :--- |
| `analysis_twfe.py` | **主要分析**。四半期データを用いたTWFE-DIDモデルを実行し、論文の主要な結論（Section 5.2）を導出します。| Section 5.2 |
| `analysis_static_did.py` | 月次データを用いた**静的DIDモデル**（OLS）を実行します。非有意な結果とモデルの限界を示したもの（Section 5.1）を再現します。| Section 5.1 |
| `analysis_robustness.py` | TWFEモデルの**ロバストネスチェック**（代替統制群、期間除外）を実行し、結果の頑健性（Section 6）を検証します。| Section 6 |
| `requirements.txt` | 実行に必要なPythonライブラリのリスト。| N/A |
| `iip_patent_industry_did_timeseries_full.csv` | **【注意】** データファイル名。このファイルは公開されていません。研究者はこの名前でデータファイルをルートディレクトリに配置する必要があります。| N/A |

## 再現手順

### 1. 依存関係のインストール

ターミナルで以下のコマンドを実行し、必要なライブラリ（`linearmodels` を含む）をインストールします。

```bash
pip install -r requirements.txt
