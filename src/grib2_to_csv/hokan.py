import os
import datetime
import numpy as np
import util
import radiation
from numba import jit


def hokankeisan(dt2, dt5, df, df_coef_atm, df_coef_rad, temp_path, logging):
    """絶対湿度の計算、日射量と大気放射量の推計
    
    補間計算を実施し、 `{path_save}/temp/{YYYYMMDDhh0000*.pkl}` に結果保存する。

    Args:
      dt2: param dt5:
      df: 気象データ
      df_coef_atm: 重回帰係数(大気放射量の推計)
      df_coef_rad: 重回帰係数(射量・大気放射量の計算)
      index_list: param temp_path: 一時ファイルの保存パス
      logging: ロガー
      dt5: 
      temp_path: 

    Returns:

    """

    date, _ = util.get_datetime_str(dt5, 2 + 9)  # JST補正（UTC+9）

    ###
    if dt2 is np.nan:  # 2ファイル分の読み込みが終わっていない1回目はパス
        pass
    else:
        for i in range(3):
            date = dt2 + datetime.timedelta(hours=i + 9)  # JST（UTC+9）に変更

            # 絶対湿度の計算、日射量と大気放射量の推計
            df_save = hokan_core(date, df, df_coef_atm, df_coef_rad)

            # 結果保存
            pickle_path = \
                os.path.join(temp_path, date.strftime(util.date_format) + '.pkl')
            df_save.to_pickle(pickle_path)
            logging.info('保存しました: {}'.format(pickle_path))

    return date


def hokan_core(date, df, df_coef_atm, df_coef_rad):
    """絶対湿度の計算、日射量と大気放射量の推計

    Args:
      date:
      df_coef_atm: 回帰係数(大気放射量の推計)
      df_coef_rad: 回帰係数(射量・大気放射量の計算)
      df: 

    Returns:
      結果のデータフレーム

    """
    d0 = date.strftime(util.date_format)

    # 重量絶対湿度
    df.loc[:, d0 + '_MR'] = func_MR(df[d0 + '_TMP'].values,
                                    df[d0 + '_RH'].values,
                                    df[d0 + '_PRES'].values)

    # 日射量・大気放射量の計算に使用する前時間との平均値を算出
    date_d1 = date + datetime.timedelta(hours=-1)
    d1 = date_d1.strftime(util.date_format)

    # <平均>気温
    TMP = (df[d0 + '_TMP'].values + df[d1 + '_TMP'].values) / 2

    # <平均>相対湿度
    RH = (df[d0 + '_RH'].values + df[d1 + '_RH'].values) / 2

    # <平均>低層雲量(CDC)  (100分率から0-1に換算)
    LCDC = (df[d0 + '_LCDC'].values + df[d1 + '_LCDC'].values) / 200

    # <平均>中層雲量(CDC)  (100分率から0-1に換算)
    MCDC = (df[d0 + '_MCDC'].values + df[d1 + '_MCDC'].values) / 200

    # <平均>高層雲量(CDC)  (100分率から0-1に換算)
    HCDC = (df[d0 + '_HCDC'].values + df[d1 + '_HCDC'].values) / 200

    # <平均>全雲量(CDC)  (100分率から0-1に換算)
    TCDC = (df[d0 + '_TCDC'].values + df[d1 + '_TCDC'].values) / 200

    # <平均>低層雲量(平均処理なし) (100分率から0-1に換算)
    LCDC5 = (df[d0 + '_LCDC5'].values + df[d1 + '_LCDC5'].values) / 200

    # <平均>低層雲量(形態係数の重みづけ平均) (100分率から0-1に換算)
    LCDC25 = (df[d0 + '_LCDC25'].values + df[d1 + '_LCDC25'].values) / 200

    # <平均>中高層雲量(平均処理なし) (100分率から0-1に換算)
    HMCDC5 = (df[d0 + '_HMCDC5'].values + df[d1 + '_HMCDC5'].values) / 200

    # <平均>中高層雲量(形態係数の重みづけ平均) (100分率から0-1に換算)
    HMCDC25 = (df[d0 + '_HMCDC25'].values + df[d1 + '_HMCDC25'].values) / 200

    # 日射量の推計
    df.loc[:, d0 + '_DSWRF_est'] = \
        radiation.func_radiation(date,
                                 df['lat'].values,
                                 df['lon'].values,
                                 HCDC,
                                 MCDC,
                                 LCDC,
                                 TCDC,
                                 TMP,
                                 RH,
                                 df_coef_rad)

    # 日射量の有無を判別して、あれば"DSWRF_msm"を[MJ/㎡]に換算
    if d0 + '_DSWRF' in df.columns:
        df.loc[:, d0 + '_DSWRF_msm'] = \
            df[d0 + '_DSWRF'].values * 3600 * 10 ** -6

    else:
        df.loc[:, d0 + '_DSWRF_msm'] = np.nan

    # 大気放射量の推計
    df.loc[:, d0 + '_Ld'] = \
        radiation.func_a_radiation(TMP,
                                   RH,
                                   LCDC5,
                                   LCDC25,
                                   HMCDC5,
                                   HMCDC25,
                                   df[d0 + '_APCP01'].values,
                                   df_coef_atm)

    # データ保存項目(計算項目)
    save_list = ['TMP',         # 気温
                 'MR',          # 重量絶対湿度
                 'DSWRF_est',   # 日射量(推計値)
                 'DSWRF_msm',   # 日射量(MSM)
                 'Ld',          # 大気放射量
                 'VGRD',        # 南北風
                 'UGRD',        # 東西風
                 'PRES',        # 地上気圧
                 'APCP01']      # 降水量

    # データ保存項目(保存時ヘッダ)
    save_name = ['TMP',         # 気温
                 'MR',          # 重量絶対湿度
                 'DSWRF_est',   # 日射量(推計値)
                 'DSWRF_msm',   # 日射量(MSM)
                 'Ld',          # 大気放射量
                 'VGRD',        # 南北風
                 'UGRD',        # 東西風
                 'PRES',        # 地上気圧
                 'APCP01']      # 降水量

    # データを保存
    exp_list = ['index'] + [d0 + '_' + s for s in save_list]
    df_save = df[exp_list]
    df_save.columns = ['index'] + save_name
    df_save.insert(0, 'date', date)

    return df_save




@jit("f8[:](f8[:],f8[:],f8[:])", nopython=True)
def func_MR(t, RH, P):
    """重量絶対湿度(SH)(kg/kg)

    Args:
      t: 温度 (C)
      RH: 相対湿度 (%)
      P: 地上気圧

    Returns:
      MR: 重量絶対湿度 (kg/kg)
    """
    # 絶対温度 (K)
    T = t + 273.15

    # 飽和水蒸気圧 (hPa) - Wexler-Hyland(ウェクスラーハイランド)の式
    eSAT = np.exp(-5800.2206 / T
                  + 1.3914993
                  - 0.048640239 * T
                  + 0.41764768 * 10 ** (-4) * T**2
                  - 0.14452093 * 10 ** (-7) * T**3
                  + 6.5459673 * np.log(T)) / 100

    # 飽和水蒸気量 (g/m3)
    aT = (217 * eSAT) / T

    # 容積絶対湿度 (g/m3)
    VH = aT * (RH / 100)

    # 重量絶対湿度(SH)(kg/kg)
    MR = VH / ((P / 100) / (2.87 * T))

    return MR