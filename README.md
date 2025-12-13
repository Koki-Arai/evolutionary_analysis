```md
# Empirical Analysis of the Evolution of Employee Invention制度  
## Evidence from Japanese Patent Microdata (1990s-2020s)

本リポジトリは、日本の職務発明制度の進化を、  
**進化ゲーム理論（Tsuneki, 2025）と整合的な実証枠組み**に基づいて検証するための  
データ構築・実証分析コードをまとめたものである。

分析は、1990年代から2020年代にかけての日本特許データを用い、  
発明者と企業の相対的生産性（β/α）と  
企業側取り分（企業名義特許割合）の関係を中心に検証する。

---

##  ディレクトリ構成

```

.
├── data/
│   ├── raw/
│   │   ├── ap_1990s.txt
│   │   ├── ap_2000s.txt
│   │   ├── ap_2010s.txt
│   │   ├── ap_2020s.txt
│   │   ├── applicant_*.txt
│   │   ├── inventor_*.txt
│   │   └── cc_*.txt
│   │
│   ├── intermediate/
│   │   ├── patent_base_1990s.parquet
│   │   ├── patent_base_2000s.parquet
│   │   ├── patent_base_2010s.parquet
│   │   └── patent_base_2020s.parquet
│   │
│   └── final/
│       └── final_empirical_panel_1990_2020.parquet
│
├── src/
│   ├── build_panel_1990s.py
│   ├── build_panel_2000s.py
│   ├── build_panel_2010s.py
│   ├── build_panel_2020s.py
│   ├── merge_panels.py
│   ├── baseline_estimations.py
│   ├── robustness_checks.py
│   └── heterogeneity_and_dynamics.py
│
├── requirements.txt
└── README.md

```

---

##  データ概要

### 特許データ（IIP/JPO）
- **ap_*.txt**  
  出願番号（ida）、出願日、IPC分類など
- **applicant_*.txt**  
  出願人（企業）情報
- **inventor_*.txt**  
  発明者情報
- **cc_*.txt**  
  特許引用データ（Forward Citations）

### 分析単位
- **エンティティ**：企業 × 技術分野（IPCセクションまたはクラス）
- **時間**：年次（1990-2020）

---

##  主要変数の定義

### 被説明変数
- **Y_firm_share**  
  企業名義特許割合（企業側取り分）

### 主要説明変数
- **beta_div_alpha (β/α)**  
  企業生産性 β と発明者平均生産性 α の比率  
  （Forward Citations に基づく特許価値で重み付け）

### 制度ショック関連
- **HighBeta_f**：改正前 β/α 上位企業ダミー  
- **DID_2015**：HighBeta_f × Post2015  
- **Time_to_Event**：2004年中村事件からの相対年

---

##  実証戦略

### 1. 基本回帰（固定効果モデル）
```

Y_fkt = γ0 + γ1 (β/α)_fkt + X_fkt′δ + μ_f + λ_t + ε_fkt

````

- 企業×技術分野固定効果
- 年固定効果
- クラスターロバスト標準誤差

### 2. 内生性対応（IV-2SLS）
- 道具変数：同一分野×年の平均 β/α（自社除外）

### 3. 制度ショック分析
- **DID（2015年改正）**
- **イベントスタディ（2004年中村事件）**

---

##  ロバストネス・精緻化

- 固定効果構造の変更（企業FEのみ／年FEのみ）
- サンプル期間の限定
- 技術分野別異質性（三重交差項）
- 分散ラグモデル（年次）

---

##  実行手順

### 1. 依存関係のインストール
```bash
pip install -r requirements.txt
````

### 2. 年代別パネル構築

```bash
python src/build_panel_1990s.py
python src/build_panel_2000s.py
python src/build_panel_2010s.py
python src/build_panel_2020s.py
```

### 3. 統合パネル作成

```bash
python src/merge_panels.py
```

### 4. ベースライン推定

```bash
python src/baseline_estimations.py
```

### 5. ロバストネス・精緻化分析

```bash
python src/robustness_checks.py
python src/heterogeneity_and_dynamics.py
```

---

##  研究上の位置付け

本分析は、

* 職務発明制度
* 分配構造
* 進化的に安定な選好

を結び付ける理論モデル（Tsuneki, 2025）を、
**長期・大規模なミクロ特許データで初めて体系的に検証**する試みである。

特に、

* IV推定による内生性の克服
* 制度ショックの動学的評価
* 異質性と移行期の不安定性

を同時に扱う点に特徴がある。

---

##  ライセンス・注意事項

* 本コードは学術研究目的での利用を想定している。
* 元データ（IIP/JPO特許データ）の再配布は禁止されているため、本リポジトリには含まれない。
* 実行には各自で正規に取得したデータが必要である。

---

## 連絡先

* **Koki Arai**
  Professor, Faculty of Business Studies
  Kyoritsu Women’s University
  Email: [koki.arai@nifty.ne.jp](mailto:koki.arai@nifty.ne.jp)

```
