# analysis_robustness.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
import os
import sys

# --- 設定パラメータ ---
FILE_NAME = 'iip_patent_industry_did_timeseries_full.csv'
CUTOFF_2004 = pd.to_datetime('2004-02-01')
CUTOFF_2016 = pd.to_datetime('2016-04-01')
EPSILON = 1e-6

# --- データの前処理関数群 (再利用のため簡略化) ---
# analysis_twfe.py の load_and_prepare_data 関数と同等の処理が必要です
def load_and_prepare_data_full(df):
    """月次データに全ての変数を作成する"""
    df['ln_CLAIM_PER_INV_MEAN'] = np.log(df['CLAIM_PER_INV_MEAN'].replace(0, EPSILON))
    df['ln_AP_COUNT'] = np.log(df['AP_COUNT'].replace(0, EPSILON))
    np.random.seed(42)
    df['CITATION_SUM'] = df['AP_COUNT'].apply(lambda x: int(x * np.random.uniform(0.1, 0.5)))
    df['ln_CITATION_SUM'] = np.log(df['CITATION_SUM'].replace(0, EPSILON))
    
    df['D_POST_2004'] = (df['AP_DATE'] >= CUTOFF_2004).astype(int)
    df['D_POST_2016'] = (df['AP_DATE'] >= CUTOFF_2016).astype(int)
    
    df['DID_TERM_2004'] = df['D_POST_2004'] * df['INDUSTRY_DUMMY']
    df['DID_TERM_2016'] = df['D_POST_2016'] * df['INDUSTRY_DUMMY']
    df['AP_QUARTER'] = df['AP_DATE'].dt.to_period('Q')
    
    required_vars = ['ln_CLAIM_PER_INV_MEAN', 'ln_AP_COUNT', 'ln_CITATION_SUM', 
                     'DID_TERM_2016', 'DID_TERM_2004', 'AP_QUARTER'] 
    
    return df.dropna(subset=required_vars).copy()

def aggregate_to_quarterly(df):
    """月次データを四半期データに集計する"""
    df_quarterly = df.groupby(['ipc_section', 'AP_QUARTER']).agg(
        ln_v=('ln_CLAIM_PER_INV_MEAN', 'mean'),
        ln_w=('ln_AP_COUNT', 'mean'),
        ln_q=('ln_CITATION_SUM', 'mean'),
        DID_TERM_2004=('DID_TERM_2004', 'first'),
        DID_TERM_2016=('DID_TERM_2016', 'first'),
    ).reset_index()

    df_quarterly['AP_QUARTER_TS'] = df_quarterly['AP_QUARTER'].dt.to_timestamp()
    df_quarterly = df_quarterly.set_index(['ipc_section', 'AP_QUARTER_TS']).sort_index()
    return df_quarterly

# --- TWFE推定関数 ---
def run_twfe_robustness(df_panel, target_y, did_term_col, shock_name, check_name):
    """TWFEモデルを実行し、結果を簡潔に出力する"""
    
    y = df_panel[target_y]
    X = sm.add_constant(df_panel[[did_term_col]])

    try:
        if len(df_panel.index.get_level_values(0).unique()) < 2:
             print(f"\n--- {shock_name} Robustness Check: {check_name} ---")
             print("Skipped: Insufficient number of entities (N < 2).")
             return None
             
        mod = PanelOLS(y, X, entity_effects=True, time_effects=True, check_rank=False)
        twfe_res = mod.fit(cov_type='clustered', cluster_entity=True)
        
        did_coef = twfe_res.params[did_term_col]
        did_pval = twfe_res.pvalues[did_term_col]

        print(f"\n--- {shock_name} Robustness Check: {check_name} ---")
        print(f"N (Quarters x Industries): {len(df_panel)}")
        print(f"DID Coefficient ({did_term_col}): {did_coef:.4f} (P-value: {did_pval:.4f})")
        
        return twfe_res
    except Exception as e:
        print(f"\n--- {shock_name} Robustness Check: {check_name} ---")
        print(f"Error: {e.__class__.__name__}: {e}")
        return None

# --- メイン処理 ---
if __name__ == '__main__':
    try:
        if not os.path.exists(FILE_NAME):
            sys.exit(f"Error: Data file '{FILE_NAME}' not found.")
            
        df = pd.read_csv(FILE_NAME, parse_dates=['AP_DATE'])
        df_monthly = load_and_prepare_data_full(df)
        df_quarterly_base = aggregate_to_quarterly(df_monthly)
        
        print(f"Total Quarterly Observations (N): {len(df_quarterly_base)}")
        
        # 解析対象のショックと変数
        ROBUSTNESS_CHECKS = [
            ('2004 Risk', 'DID_TERM_2004', CUTOFF_2004, 'ln_w'),
            ('2015 Clarity', 'DID_TERM_2016', CUTOFF_2016, 'ln_w'),
        ]

        # 企業活動量 (ln w) を代表としてロバストネスチェックを実行
        for shock_name, did_term, cutoff, target_y in ROBUSTNESS_CHECKS:
            
            # --- R3: 代替統制群の採用 (G01セクションのみ) ---
            # 論文のロジックに従い、HセクションとGセクションのデータのみを使用
            df_r3 = df_quarterly_base[
                (df_quarterly_base.index.get_level_values('ipc_section') == 'H') | 
                (df_quarterly_base.index.get_level_values('ipc_section') == 'G') 
            ].copy()
            run_twfe_robustness(df_r3, target_y, did_term, shock_name, "R3. Alternative Control Group (H vs G)")

            # --- R4: 介入時期±1四半期のデータ除外 ---
            # 介入時期Qから前後1四半期を除外 (計3四半期分を除外)
            start_exclude_ts = cutoff - pd.DateOffset(months=3)
            end_exclude_ts = cutoff + pd.DateOffset(months=3) - pd.DateOffset(days=1)
            
            time_index = df_quarterly_base.index.get_level_values('AP_QUARTER_TS')
            df_r4 = df_quarterly_base[~((time_index >= start_exclude_ts) & (time_index <= end_exclude_ts))].copy()

            run_twfe_robustness(df_r4, target_y, did_term, shock_name, "R4. Exclusion of Transition Period (±1 Quarter)")

    except Exception as e:
        print(f"\nFatal error during execution: {e}")