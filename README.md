# Employee Inventions: Evolutionary Equilibrium and Institutional Shocks (Replication Package)

本リポジトリは、進化ゲーム理論に基づき、日本の職務発明制度における二つの主要な制度的ショック（2004年の中村事件判決と2015年特許法改正）がR&Dインセンティブに与えた異質な影響を分析した論文の再現パッケージです。

この研究は、法制度の変更に対する企業の反応が、ショックの性質（「リスク」対「明確性」）によって大きく異なることを実証しています。

## 論文情報

* **論文タイトル**: Evolving Ownership and Employee Inventions: Equilibrium and Shocks Revealed by Japanese Patent Data
* **主要な発見**: 
    1. どちらのショックも、発明者インセンティブと特許の質（ln v, ln q）を**一貫して抑制**した。
    2. 企業活動量（ln w）については、**リスクショック（2004年）で促進**（防御的拡張）、**明確性シフト（2015年）で抑制**（合理化・選別）という、**「リスク対明確性パラドックス」**を発見した。

## ファイル構成

| ファイル名 | 説明 |
| :--- | :--- |
| `analysis_twfe.py` | メインの計量経済学分析コード。四半期データを用いたTWFE-DIDモデルを実行します。|
| `requirements.txt` | 実行に必要なPythonライブラリのリスト。|
| `README.md` | 本ファイル。プロジェクトの説明と実行手順。|
| `iip_patent_industry_did_timeseries_full.csv` | **【注意】** データファイル名。このファイルは公開されていません。研究者はこの名前でデータファイルをルートディレクトリに配置する必要があります。|

## 再現手順

### 1. 依存関係のインストール

ターミナルで以下のコマンドを実行し、必要なライブラリをインストールします。

```bash
pip install -r requirements.txt
