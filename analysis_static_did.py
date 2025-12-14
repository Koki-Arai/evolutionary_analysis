# analysis_static_did.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os
import sys

# --- 設定パラメータ ---
FILE_NAME = 'iip_patent_industry_did_timeseries_full.csv'
CUTOFF_2004 = pd.to_datetime('2004-02-01')
CUTOFF_2016 = pd.to_datetime('2016-04-01')
EPSILON = 1e-6

# --- データの前処理と変数作成 ---
def load_and_prepare_data(df):
    """月次データを読み込み、対数変数とDID項、タイムトレンドを作成する"""
    
    # 応答変数と対数変換 (ln Y)
    df['ln_CLAIM_PER_INV_MEAN'] = np.log(df['CLAIM_PER_INV_MEAN'].replace(0, EPSILON))
    df['ln_AP_COUNT'] = np.log(df['AP_COUNT'].replace(0, EPSILON))
    # ln q (ln_CITATION_MEAN) はゼロ値問題のため、この静的DID分析では省略
    
    # D_POST 変数 (時間ダミー)
    df['D_POST_2004'] = (df['AP_DATE'] >= CUTOFF_2004).astype(int)
    df['D_POST_2016'] = (df['AP_DATE'] >= CUTOFF_2016).astype(int)
    
    # DID_TERM 変数
    df['DID_TERM_2004'] = df['D_POST_2004'] * df['INDUSTRY_DUMMY']
    df['DID_TERM_2016'] = df['D_POST_2016'] * df['INDUSTRY_DUMMY']
    
    # TIME_TREND 変数 (Model B で使用)
    # ipc_section ごとの通し番号 (月次の連番)
    df['TIME_TREND'] = df.groupby('ipc_section').cumcount() + 1 
    
    required_vars = ['ln_CLAIM_PER_INV_MEAN', 'ln_AP_COUNT', 
                     'INDUSTRY_DUMMY', 'D_POST_2004', 'D_POST_2016', 
                     'DID_TERM_2004', 'DID_TERM_2016', 'TIME_TREND'] 
    
    return df.dropna(subset=required_vars).copy()

# --- 静的DID推定関数 ---
def run_static_did(df_data, target_y, did_term_col, has_trend, title, shock_name):
    """月次静的DIDモデル (OLS) を実行する関数 (Section 4.1)"""
    
    # 基本のDIDモデル式: ln Y ~ INDUSTRY_DUMMY + D_POST + DID_TERM
    formula = f"{target_y} ~ INDUSTRY_DUMMY + {did_term_col.replace('DID_TERM', 'D_POST')} + {did_term_col}"
    
    # Model B (2015 Revision) の場合のみ TIME_TREND を追加
    if has_trend:
        formula += " + TIME_TREND"
        
    try:
        # ロバスト標準誤差 (HC1) を使用
        model = ols(formula, data=df_data).fit(cov_type='HC1')
        
        print(f"\n--- {shock_name} Static DID ({'Model B: Trend' if has_trend else 'Model A: Static'}) for {target_y} ---")
        print(model.summary())
        
        return model
    except Exception as e:
        print(f"\n--- {shock_name} Static DID Estimation Failed for {target_y} ---")
        print(f"Error: {e}")
        return None

# --- メイン処理 ---
if __name__ == '__main__':
    try:
        # データの読み込み
        if not os.path.exists(FILE_NAME):
            sys.exit(f"Error: Data file '{FILE_NAME}' not found. Please place the data file in the root directory.")
            
        df = pd.read_csv(FILE_NAME, parse_dates=['AP_DATE'])
        
        # データの準備
        df_clean = load_and_prepare_data(df)
        
        print(f"Total Monthly Observations (N): {len(df_clean)}")
        
        # 2004年 中村事件 - Model A: Static DID (トレンドなし)
        # 応答変数: ln v, ln w
        run_static_did(df_clean, 'ln_CLAIM_PER_INV_MEAN', 'DID_TERM_2004', False, 'Model A', '2004 Nakamura')
        run_static_did(df_clean, 'ln_AP_COUNT', 'DID_TERM_2004', False, 'Model A', '2004 Nakamura')
        
        # 2015年改正 - Model B: DID with Time Trend (トレンドあり)
        # 応答変数: ln v, ln w
        run_static_did(df_clean, 'ln_CLAIM_PER_INV_MEAN', 'DID_TERM_2016', True, 'Model B', '2015 Revision')
        run_static_did(df_clean, 'ln_AP_COUNT', 'DID_TERM_2016', True, 'Model B', '2015 Revision')

    except Exception as e:
        print(f"\nFatal error during execution: {e}")