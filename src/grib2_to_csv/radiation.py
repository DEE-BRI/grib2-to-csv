from numba import jit
import numpy as np
import datetime
import pandas as pd
from math import *

def func_radiation(date, lat, lon, HCDC, MCDC, LCDC, TCDC, TMP, RH,
                   df_coef_rad):
    """日射量の推計

    Args:
      date: 
      lat: 緯度
      lon: 経度
      HCDC: 高層雲量 (-)
      MCDC: 中層雲量 (-)
      LCDC: 低層雲量 (-)
      TCDC: 全雲量 (-)
      TMP: 気温
      RH: 相対湿度
      df_coef_rad: 

    Returns:

    """
    # 大気外日射量とエアマスの計算
    IHO, AM = func_IHO_AM(date, lat, lon)

    # 計算用のdfを作成
    df_rad = \
        pd.DataFrame(
            list(zip(HCDC, MCDC, LCDC, TCDC, AM, TMP, RH, IHO)),
            columns=['HCDC', 'MCDC', 'LCDC', 'TCDC', 'AM', 'TMP', 'RH', 'IHO'])

    # エアマスでクラス分け
    AM_max = max(df_rad['AM'].max() + 0.1, 2.1)
    AM_cut = [-0.1, 1.5, 2.0, AM_max]
    df_rad['AM_rank'] = pd.cut(df_rad['AM'],
                               AM_cut,
                               right=False,
                               precision=6,
                               labels=['A1', 'A2', 'A3'])

    # 全雲量でクラス分け
    TC_cut = [-0.01, 0.02, 0.5, 0.98, 1.01]
    df_rad['TC_rank'] = pd.cut(df_rad['TCDC'],
                               TC_cut,
                               right=False,
                               precision=6,
                               labels=['C1', 'C2', 'C3', 'C4'])

    # クラスごとにグループ分けし、それぞれを計算
    df_group = df_rad.groupby(['AM_rank', 'TC_rank'])
    for name, group in df_group:
        g_index = group.index.values

        a, b, c, d, e, f, g = oder_select_rad(df_coef_rad, ''.join(name))
        df_rad.loc[g_index, 'DSWRF_est'] = \
            func_TH(df_rad.loc[g_index, 'HCDC'].values,
                    df_rad.loc[g_index, 'MCDC'].values,
                    df_rad.loc[g_index, 'LCDC'].values,
                    df_rad.loc[g_index, 'AM'].values,
                    df_rad.loc[g_index, 'TMP'].values,
                    df_rad.loc[g_index, 'RH'].values,
                    a, b, c, d, e, f, g,
                    df_rad.loc[g_index, 'IHO'].values)

    DSWRF_est = df_rad["DSWRF_est"].values
    df_rad = None
    return DSWRF_est


def func_a_radiation(TMP, RH, LCDC5, LCDC25, HMCDC5, HMCDC25, APCP01,
                     df_coef_atm):
    """大気放射量の推計

    Args:
      TMP: 気温
      RH: 相対湿度
      LCDC5: 低層雲量(平均処理なし)
      LCDC25: 低層雲量(形態係数の重みづけ平均)
      HMCDC5: 中高層雲量(平均処理なし)
      HMCDC25: 中高層雲量(形態係数の重みづけ平均)
      APCP01: 降水量
      df_coef_atm: 

    Returns:

    """

    # 大気放射量の補間計算
    e = func_e(TMP, RH)

    # 計算用のdfを作成
    df_rad = \
        pd.DataFrame(
            list(zip(LCDC5, LCDC25, HMCDC5, HMCDC25, e, APCP01)),
            columns=['LCDC5', 'LCDC25', 'HMCDC5', 'HMCDC25', 'e', 'APCP01'])

    df_rad.loc[df_rad['APCP01'] == 0, 'PP'] = 'N'
    df_rad.loc[df_rad['APCP01'] != 0, 'PP'] = 'P'

    # 降雨なしの条件
    if 'N' in df_rad['PP'].values:
        u, v, w, x = oder_select_atm(df_coef_atm, 'N')
        df_rad.loc[df_rad['APCP01'] == 0, 'C'] = func_C(df_rad['HMCDC25'],
                                                        df_rad['LCDC25'],
                                                        df_rad['e'],
                                                        u, v, w, x)
    # 降雨ありの条件
    if 'P' in df_rad['PP'].values:
        u, v, w, x = oder_select_atm(df_coef_atm, 'P')
        df_rad.loc[df_rad['APCP01'] != 0, 'C'] = func_C(df_rad['HMCDC5'],
                                                        df_rad['LCDC5'],
                                                        df_rad['e'],
                                                        u, v, w, x)

    Ld = func_Ld(TMP, e, df_rad['C'].values)
    df_rad = None

    return Ld




@jit(nopython=True)
def func_TH(clu_data, clm_data, cll_data, AM, TT, RH,
            a, b, c, d, e, f, g, IHO):
    """

    Args:
      clu_data: 
      clm_data: 
      cll_data: 
      AM: 
      TT: 
      RH: 
      a: 
      b: 
      c: 
      d: 
      e: 
      f: 
      g: 
      IHO: 

    Returns:

    """
    return (a * clu_data
            + b * clm_data
            + c * cll_data
            + d * AM
            + e * TT
            + f * RH
            + g) * IHO


@jit("f8[:](f8[:],f8[:],f8[:])", nopython=True)
def func_Ld(TMP, e, C):
    """

    Args:
      TMP: 気温
      e: 
      C: 

    Returns:

    """
    TMP_abs = TMP + 273.15

    # ステファンボルツマン定数
    SG = 5.67*10**(-8)

    # 露点温度
    Tdew = 237.3 * np.log10(e / 6.11) / (7.5 - np.log10(e / 6.11))
    x = 0.0315 * Tdew - 0.1836
    Ldf = (0.74 + 0.19 * x + 0.07 * x ** 2) * SG * TMP_abs ** 4

    return SG * TMP_abs ** 4 * (1 - (1 - Ldf / (SG * TMP_abs ** 4)) * C)




def func_IHO_AM(day_date, phi, lon):
    """

    Args:
      day_date: 
      phi: 
      lon: 経度

    Returns:

    """
    year = datetime.date(day_date.year, 1, 1)
    t1 = day_date.date()
    TM = day_date.hour
    J0 = 4.392  # 太陽定数[MH/m²h]
    count = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    Nday = t1 - year  # 年間通日
    nday = int(Nday.days) + 1
    DY = int(t1.year)
    Tm = TM  # 標準時
    dlt0 = radians(-23.4393)  # 冬至の日赤緯
    n = DY - 1968
    d0 = 3.71 + 0.2596 * n - int((n + 3) / 4)  # 近日点通過日
    m = 360 * (nday - d0) / 365.2596  # 平均近点離角
    eps = 12.3901 + 0.0172 * (n + m / 360)  # 近日点と冬至点の角度
    v = m + 1.914 * sin(radians(m)) + 0.02 * sin(radians(2 * m))  # 真近点離角
    veps = radians(v + eps)
    Et = (m - v) - degrees(atan(0.043 * sin(2 * veps) /
                                (1 - 0.043 * cos(2 * veps))))  # 近時差
    sindlt = cos(veps) * sin(dlt0)  # 赤緯の正弦
    cosdlt = (fabs(1 - sindlt ** 2)) ** 0.5  # 赤緯の余弦
    INO = J0 * (1 + 0.033 * cos(radians(v)))  # IN0 大気外法線面日射量

    lons = 135  # 標準時の地点の経度
    phirad = np.radians(phi)  # 緯度

    IHO = pd.DataFrame()  # IHOの容器
    AM = pd.DataFrame()  # AMの容器

    for j in count:
        tm = Tm - j
        t = 15 * (tm - 12) + (lon - lons) + Et  # 時角
        trad = np.radians(t)
        sinh = \
            pd.DataFrame(
                np.sin(phirad) * sindlt
                + np.cos(phirad) * cosdlt * np.cos(trad),
                columns=['sinh'])  # 太陽高度角の正弦

        if (sinh['sinh'] >= 0).sum() == 0:
            sinh['IHO'] = 0.0
            sinh['AM'] = 0.0

        else:
            sinh.loc[sinh['sinh'] >= 0.0, 'IHO'] = INO * sinh['sinh']
            sinh.loc[sinh['sinh'] < 0.0, 'IHO'] = 0.0

            sinh.loc[sinh['sinh'] >= 0.0, 'AM'] = \
                1 / (sinh['sinh'] + 0.15 *
                     (np.degrees(np.arcsin(sinh['sinh'])) + 3.885) ** (-1.253))
            sinh.loc[sinh['sinh'] < 0.0, 'AM'] = 0.0

        IHO[str(j)] = sinh['IHO']
        AM[str(j)] = sinh['AM']

    return IHO.mean(axis=1).values, AM.mean(axis=1).values


def oder_select_rad(df, oder):
    """

    Args:
      df: 
      oder: 

    Returns:

    """
    return df.at[oder, 'a'], \
           df.at[oder, 'b'], \
           df.at[oder, 'c'], \
           df.at[oder, 'd'], \
           df.at[oder, 'e'], \
           df.at[oder, 'f'], \
           df.at[oder, 'g']


def oder_select_atm(df, oder):
    """

    Args:
      df: 
      oder: 

    Returns:

    """
    return df.at[oder, 'u'], \
           df.at[oder, 'v'], \
           df.at[oder, 'w'], \
           df.at[oder, 'x']


def func_C(HMCDC, LCDC, e, u, v, w, x):
    """

    Args:
      HMCDC: 中高層雲量(25㎞移動平均)
      LCDC: 低層雲量(25㎞移動平均)
      e: 
      u: 
      v: 
      w: 
      x: 

    Returns:

    """
    return 1 + (u*LCDC + v*LCDC*e + w*HMCDC + x*HMCDC * e)



@jit("f8[:](f8[:],f8[:])", nopython=True)
def func_e(TMP, RH):
    """

    Args:
      TMP: 気温
      RH: 相対湿度

    Returns:

    """
    TMP_abs = TMP + 273.15
    eSAT = np.exp(-5800.2206 / TMP_abs
                  + 1.3914993
                  - 0.048640239 * TMP_abs
                  + 0.41764768 * 10 ** (-4) * TMP_abs**2
                  - 0.14452093 * 10 ** (-7) * TMP_abs**3
                  + 6.5459673 * np.log(TMP_abs)) / 100
    return eSAT * RH / 100
