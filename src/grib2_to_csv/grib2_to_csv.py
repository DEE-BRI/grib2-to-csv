import pandas as pd
from osgeo import gdal
import logging

from scipy.ndimage import uniform_filter

from numba import jit
import numpy as np
from functools import lru_cache
from math import *

import datetime
import glob
import os
import pathlib
import re
import shutil

import numpy as np
import pandas as pd
from osgeo import gdal
from scipy import ndimage, signal

import util
import hokan


def get_MSM_index_lat_lon():
    """DFのインデックスとしてMSMの座標と緯度経度を読み込む

    Args:

    Returns:
      DataFrame: MSMの座標と緯度経度

    """
    MSM_lat = np.arange(505)  # 緯度方向の格子点数505
    MSM_lon = np.arange(481)  # 経度方向の格子点数481

    lat_mesh, lon_mesh = np.meshgrid(MSM_lat, MSM_lon)  # 緯度経度格子点の組み合わせ

    lat_mesh = lat_mesh.T.ravel()  # 一次元データへ変換

    lat = pd.Series(47.6 - (lat_mesh * 0.05))
    lat.name = 'lat'

    lon_mesh = lon_mesh.T.ravel()  # 一次元データへ変換

    lon = pd.Series(120.0 + (lon_mesh * 0.0625))
    lon.name = 'lon'

    lat_mesh = pd.Series(lat_mesh).astype(str)  # pandas.Seriesへ変更
    lon_mesh = pd.Series(lon_mesh).astype(str)  # pandas.Seriesへ変更

    # 緯度と経度の座標番号を'-'で組み合わせ
    MSM_mesh_index = lat_mesh.str.cat(lon_mesh, sep='-')
    MSM_mesh_index.name = 'index'  # カラム名をindexに変更

    MSM_index_lat_lon = pd.concat([MSM_mesh_index, lat, lon], axis=1)

    return MSM_index_lat_lon


def get_MSM_list(df_meshcode):
    """

    Args:
      df_meshcode([type]): [description]

    Returns:
      [type]: [description]

    """
    df_meshcode_L = len(df_meshcode)

    lat_unit = 0.05  # MSMの緯度間隔
    lon_unit = 0.0625  # MSMの経度間隔

    # メッシュコードから緯度経度を計算(中心ではなく南西方向の座標が得られる)
    y1 = df_meshcode['meshcode'].astype(str).str[:2].astype(int).values
    x1 = (df_meshcode['meshcode'].astype(str).str[2] +
          df_meshcode['meshcode'].astype(str).str[3]).astype(int).values
    y2 = df_meshcode['meshcode'].astype(str).str[4].astype(int).values
    x2 = df_meshcode['meshcode'].astype(str).str[5].astype(int).values
    y3 = df_meshcode['meshcode'].astype(str).str[6].astype(int).values
    x3 = df_meshcode['meshcode'].astype(str).str[7].astype(int).values

    # 南西方向の座標からメッシュ中心の緯度を算出
    lat = ((y1 * 80 + y2 * 10 + y3) * 30 / 3600) + 15 / 3600

    # 南西方向の座標からメッシュ中心の経度を算出
    lon = (((x1 * 80 + x2 * 10 + x3) * 45 / 3600) + 100) + 22.5 / 3600

    # メッシュ周囲のMSM位置（緯度経度）と番号（北始まり0～、西始まり0～）の取得

    # 北は切り上げ
    lat_N = np.ceil(lat / lat_unit) * lat_unit
    MSM_N = np.round((47.6 - lat_N) / lat_unit).astype(int)

    # 南は切り下げ
    lat_S = np.floor(lat / lat_unit) * lat_unit
    MSM_S = np.round((47.6 - lat_S) / lat_unit).astype(int)

    # 西は切り下げ
    lon_W = np.floor(lon / lon_unit) * lon_unit
    MSM_W = np.round((lon_W - 120) / lon_unit).astype(int)

    # 東は切り上げ
    lon_E = np.ceil(lon / lon_unit) * lon_unit
    MSM_E = np.round((lon_E - 120) / lon_unit).astype(int)

    # 保存用のDFを作成
    df = pd.DataFrame()

    MSM_no_UD = np.concatenate([MSM_N, MSM_N, MSM_S, MSM_S])
    MSM_no_LR = np.concatenate([MSM_W, MSM_E, MSM_W, MSM_E])

    df['MSM_no_UD'] = MSM_no_UD
    df['MSM_no_LR'] = MSM_no_LR

    df = df.drop_duplicates().reset_index(drop=True)

    df['index'] = df['MSM_no_UD'].astype(str) \
        + '-' \
        + df['MSM_no_LR'].astype(str)

    return df['index'].values


def pd_read_pikle_map(file_list):
    """Pickleされたデータフレーム群を1つに結合する

    Args:
      file_list: Pickeされたデータフレームのファイル一覧

    Returns:
      結合されたデータフレーム

    """
    df = pd.concat(map(pd_read_pikle, file_list))
    return df


def pd_read_pikle(file_path):
    """

    Args:
      file_path: 

    Returns:

    """
    # Load pickled pandas object (or any object) from file.
    df = pd.read_pickle(file_path)

    # Reset the index of the DataFrame, and use the default one instead
    # Do not try to insert index into dataframe columns.
    # This resets the index to the default integer index.
    df.reset_index(drop=True)

    return df


def weather_selester(dataset, weather, fc_time, band_list):
    """

    Args:
      dataset: 
      weather: 
      fc_time: 
      band_list: 

    Returns:

    """
    data = np.nan
    fc_time = str(int(fc_time * 3600)) + ' sec'
    for band in band_list:
        d = dataset.GetRasterBand(int(band)).GetMetadata()
        if weather in d.values() and fc_time in d.values():
            data = dataset.GetRasterBand(int(band)).ReadAsArray()
            band_list.remove(band)
            break

        elif '14400 sec' in d.values():
            break

        else:
            pass

    return data, band_list


def in_series(ndata):
    """

    Args:
      ndata: 

    Returns:

    """
    return pd.Series(ndata.ravel())


@lru_cache(maxsize=1000)
def calc_VF(l, b, d):
    """大気放射量計算用の形態係数等の算出

    Args:
      l: 
      b: 
      d: 

    Returns:

    """
    return 1 / (2 * pi) \
           * (l / (sqrt(d ** 2 + l ** 2))
              * atan(b / sqrt(d ** 2 + l ** 2))
              + b / (sqrt(d ** 2 + b ** 2))
              * atan(l / (sqrt(d ** 2 + b ** 2))))


@lru_cache(maxsize=1000)
def viewfactor(H):
    """

    Args:
      H: 

    Returns:

    """
    VF = np.zeros(25)

    x1 = [12.5, 7.5, 2.5, 7.5, 12.5] * 5
    x2 = [7.5, 2.5, 0, 2.5, 7.5] * 5
    y1 = [12.5] * 5 + [7.5] * 5 + [2.5] * 5 + [7.5] * 5 + [12.5] * 5
    y2 = [7.5] * 5 + [2.5] * 5 + [0] * 5 + [2.5] * 5 + [7.5] * 5
    N = [4, 4, 3, 4, 4] * 2 + [2, 2, 1, 2, 2] + [4, 4, 3, 4, 4] * 2

    for num in range(len(N)):
        if N[num] == 1:
            VF[num] = calc_VF(y1[num], x1[num], H) * 4

        if N[num] == 2:
            VF[num] = calc_VF(y1[num], x1[num], H) * 2 \
                      - calc_VF(y1[num], x2[num], H) * 2

        if N[num] == 3:
            VF[num] = calc_VF(y1[num], x1[num], H) * 2 \
                      - calc_VF(y2[num], x1[num], H) * 2

        if N[num] == 4:
            VF[num] = calc_VF(y1[num], x1[num], H) \
                      - calc_VF(y2[num], x1[num], H) \
                      - calc_VF(y1[num], x2[num], H) \
                      + calc_VF(y2[num], x2[num], H)

    return VF.reshape(5, 5) / VF.sum()




def hokanloop(df, date_end, df_coef_atm, df_coef_rad, file_list, temp_path,
              start_date, logger):
    """補間計算ループ

    Args:
      date_2: param date_end:
      df: param df_coef_atm: 回帰係数(大気放射量の推計)
      df_coef_rad: 回帰係数(射量・大気放射量の計算)
      file_list: param temp_path: MSMファイルの保存パス
      start_date: param logger: ロガー
      date_end: 
      df_coef_atm: 
      temp_path: 
      logger: 

    Returns:

    """

    utc_jst = datetime.timedelta(hours=9)

    dt2 = np.nan

    # 3時間ごとのdatetimeのリストを作成
    date_list = []
    for loop in range(len(file_list)):

        # 計算対象日時 の計算
        if dt2 is np.nan:  # 1回目は回避
            dt2 = date_end
            date_list.append(dt2)

            if dt2 <= start_date:
                break

        else:  # 計算済み部分の3時間前のファイルを読み込み
            dt5 = dt2 + datetime.timedelta(hours=-3)
            date_list.append(dt5)

            if dt2 <= start_date:
                break

            dt2 = dt5  # 前回読み込んだファイル（3～5時間）を0～2時間のファイルとする

    # 3時間ごとに計算, 最終から逆順
    dt2 = np.nan
    for dt5 in date_list:
        # ログ出力
        logging.info((dt5 + utc_jst).strftime(util.date_format) + '(JST)を計算しています')

        # GRIB2読込ファイル名の確定
        date_str = dt5.strftime(util.date_format)
        file_end_list = [s for s in file_list if date_str in s]
        if len(file_end_list) == 0:
            logger.error('計算対象のGRIB2ファイルを見つけられませんでした。{}'.format(date_str))
            exit(1)

        # GRIB2データセットの読み込み
        grib2_file_path = file_end_list[0]
        grib2_dataset = gdal.Open(grib2_file_path, gdal.GA_ReadOnly)

        # GRIB2データセットから気象要素を取り出してDFに格納
        load_grib2_to_dataframe(grib2_dataset, dt5, df)

        # 絶対湿度の計算、日射量と大気放射量の推計を実施し、結果をpickleで保存
        date = hokan.hokankeisan(dt2, dt5, df, df_coef_atm, df_coef_rad,
                                 temp_path, logging)

        # 不要なデータをdfから削除
        df = clean_df(dt5, df, logging)

        dt2 = dt5  # 前回読み込んだファイル（3～5時間）を0～2時間のファイルとする

    logger.info('補間計算が終了しました')



def clean_df(date_5, df, logging):
    """不要なデータをdfから削除

    Args:
      date_5: param df:
      logging: return:
      df: 

    Returns:

    """
    exc_list = []
    columns = df.columns.values
    for i in range(3):
        date = date_5 + datetime.timedelta(hours=i + 9)  # JST（UTC+9）に変更
        date_str = date.strftime(util.date_format)
        exc_list += [s for s in columns if date_str in s]
    exc_list = ['lon', 'lat', 'index'] + exc_list

    logging.debug('データフレームのゴミ削除: 残すカラム{}'.format(exc_list))

    df = df[exc_list].sort_index(axis=1, ascending=False)
    return df


def load_grib2_to_dataframe(dataset, date, df):
    """データセットから気象要素を取り出してDFに格納

    Args:
      dataset: GRIB2データセット
      date: GRIB2データセットの年月日日時
      df: 格納先のデータフレーム

    Returns:

    """

    # 気象項目
    weather_set = ['PRES',      # 地上気圧
                   'UGRD',      # 東西風
                   'VGRD',      # 南北風
                   'TMP',       # 気温
                   'RH',        # 相対湿度
                   'APCP01',    # 降水量
                   'DSWRF']     # 日射量

    # GRIB2データのバンド番号のリスト ex) [0, 1, 2, ... , 9]
    band_list = list(np.arange(1, dataset.RasterCount + 1))

    index_list = df.index

    for fc_time in range(3):
        for weather in weather_set:
            # 降水量と日射量は予報時刻と1時間ずれる JST補正（UTC+9） or JST補正（UTC+10）
            timedifference = 10 if weather in ["APCP01", 'DSWRF'] else 9
            _, date_str = util.get_datetime_str(date, fc_time + timedifference)
            data_temp, band_list = \
                weather_selester(dataset, weather, fc_time, band_list)

            # 日射量がなかった場合に回避
            if data_temp is np.nan:
                pass

            else:
                df[date_str + '_' + weather] = \
                    in_series(data_temp).loc[index_list]

        # 雲量を読み込んで、日射量・大気放射量計算用の雲量を計算してDFに格納
        _, date_str = util.get_datetime_str(date, fc_time + 9)    # JST補正（UTC+9）

        # 低層雲量の読み込み
        LCDC_temp, band_list = \
            weather_selester(dataset, 'LCDC', fc_time, band_list)

        # 平均処理なし
        df[date_str + '_LCDC5'] = in_series(LCDC_temp).loc[index_list]

        # 25㎞移動平均
        df[date_str + '_LCDC'] = \
            in_series(uniform_filter(LCDC_temp, size=5, mode='constant')) \
            .loc[index_list]

        # 形態係数の重みづけ平均
        df[date_str + '_LCDC25'] = \
            in_series(signal.correlate(LCDC_temp, viewfactor(0.5), 'same')) \
            .loc[index_list]

        # 中層雲量の読み込み
        MCDC_temp, band_list = \
            weather_selester(dataset, 'MCDC', fc_time, band_list)
        df[date_str + '_MCDC'] = \
            in_series(uniform_filter(MCDC_temp, size=5, mode='constant')) \
            .loc[index_list]

        # 高層雲量の読み込み
        HCDC_temp, band_list = \
            weather_selester(dataset, 'HCDC', fc_time, band_list)
        df[date_str + '_HCDC'] = \
            in_series(uniform_filter(HCDC_temp, size=5, mode='constant')) \
            .loc[index_list]

        # 全雲量の読み込み
        TCDC_temp, band_list = \
            weather_selester(dataset, 'TCDC', fc_time, band_list)
        df[date_str + '_TCDC'] = \
            in_series(uniform_filter(TCDC_temp, size=5, mode='constant')) \
            .loc[index_list]

        # 中高層雲量の計算
        HMCDC_temp = TCDC_temp - LCDC_temp
        HMCDC_temp = np.where(HMCDC_temp < 0, 0, HMCDC_temp)

        # 平均処理なし
        df[date_str + '_HMCDC5'] = in_series(HMCDC_temp).loc[index_list]

        # 形態係数の重みづけ平均
        df[date_str + '_HMCDC25'] = \
            in_series(signal.correlate(HMCDC_temp, viewfactor(2.0), 'same')) \
            .loc[index_list]



def init(args):
    """初期化処理

    Args:
      args: コマンドライン引数

    Returns:

    """
    # GDALで気温を摂氏で読み込む
    gdal.SetConfigOption('GRIB_NORMALIZE_UNITS', 'YES')

    # ログレベル指定
    logging.basicConfig(level=logging.DEBUG)

    # 一時データの保存パス
    temp_path = os.path.join(args.path_save, 'temp')

    # ロガーの作成
    logger = logging.getLogger(__name__)

    # 一時データのディレクトリの作成
    os.makedirs(temp_path, exist_ok=True)
    logger.info('一時ディレクトリの作成: {}'.format(temp_path))

    # 重回帰係数の読み込み
    logger.info('重回帰係数(射量・大気放射量の計算)の読込: {}'.format(args.coef_rad_file))
    df_coef_rad = pd.read_csv(args.coef_rad_file, index_col=0)

    logger.info('重回帰係数(大気放射量の推計)の読込: {}'.format(args.coef_rad_file))
    df_coef_atm = pd.read_csv(args.coef_atm_file, index_col=0)

    # メッシュコードリストの読み込み
    logger.info('メッシュコードリストの読込: {}'.format(args.path_meshcode))
    df_meshcode = pd.read_csv(args.path_meshcode)

    # 計算期間
    start_date = datetime.datetime.strptime(
        args.start + '0000', util.date_format) \
        + datetime.timedelta(hours=-9)
    end_date = pd.Timestamp(
        datetime.datetime.strptime(args.end + '0000', util.date_format))\
        .ceil('3H') \
        + datetime.timedelta(hours=-9)

    # 計算対象のGRIB2ファイルをリストとして取得

    # GRIB2ファイルの一覧
    grib2_path = os.path.join(args.path_open, '*.bin')
    grib2_file_list = sorted(glob.glob(grib2_path), reverse=True)
    logger.debug('GRIB2ファイルの一覧')
    logger.debug('------------------------------------------------')
    for file in grib2_file_list:
        logger.debug('{}'.format(file))
    logger.debug('------------------------------------------------')

    return {
        #
        "path_open": args.path_open,

        #
        "path_save": args.path_save,

        # 一時データの保存パス
        "temp_path": temp_path,

        "start_date": start_date,
        # "date_end": date_end,
        "date_end": end_date,

        # GRIB2ファイルの一覧
        "grib2_file_list": grib2_file_list,

        # ロガー
        "logger": logger,

        # 重回帰係数(射量・大気放射量の計算)
        "df_coef_rad": df_coef_rad,

        # 重回帰係数(大気放射量の推計)
        "df_coef_atm": df_coef_atm,

        # メッシュコードリスト
        "df_meshcode": df_meshcode
    }


def main():
    """ """
    import argparse

    # コマンドライン引数の処理
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", help='(JST)', default='2011010100')
    parser.add_argument("--end", help='(JST)', default='2021010100')
    parser.add_argument(
        "--path_open",
        default='.{0}bin{0}'.format(os.path.sep))
    parser.add_argument(
        "--path_save",
        default='.{0}msm{0}'.format(os.path.sep))
    parser.add_argument(
        "--coef_rad_file",
        help='重回帰係数(日射量・大気放射量の計算)',
        default='.{0}data{0}coef_rad_20210125.csv'.format(os.path.sep))
    parser.add_argument(
        "--coef_atm_file",
        help='重回帰係数(大気放射量の推計)',
        default='.{0}data{0}coef_arad_20210125.csv'.format(os.path.sep))
    parser.add_argument(
        "--path_meshcode",
        help='計算対象のメッシュコードリストの保存場所',
        default='.{0}data{0}1km_meshcode_all.csv'.format(os.path.sep))
    args = parser.parse_args()

    # 初期化
    conf = init(args)
    logger = conf['logger']

    # DFのインデックスとしてMSMの座標と緯度経度を読み込む
    df_index = get_MSM_index_lat_lon()

    # メッシュデータの計算に必要なMSM座標の計算
    MSM_list = sorted(get_MSM_list(conf['df_meshcode']))

    # メッシュデータの計算に必要なMSM座標の抜粋
    df = df_index[df_index['index'].isin(MSM_list)].copy()

    # 補間計算ループ
    hokanloop(df,
              conf['date_end'], conf['df_coef_atm'], conf['df_coef_rad'],
              conf['grib2_file_list'], conf['temp_path'],
              conf['start_date'], logger)

    # ファイルの分割
    split_files(MSM_list, conf['temp_path'], logger)

    # ファイルの結合
    combine_files(MSM_list, conf['temp_path'], conf['path_save'], logger)

    logger.info("全ての作業が終了しました。")


def split_files(MSM_list, temp_path, logger):
    """ファイルの分割

    Args:
      MSM_list: MSMのリスト
      logger: ロガー
      temp_path: 分割ファイルがあるディレクトリ

    Returns:

    """

    # 読み込むファイルサイズの関係で10日分（時間）ごとに分割
    data_row_max = 2400

    # 保存されたPickelファイルの確認
    file_list = sorted(glob.glob(os.path.join(temp_path, '*.pkl')))

    # 部活ファイル数の計算
    file_count = int(np.ceil(len(file_list) / data_row_max))

    logger.info('MSM座標ごとにデータを保存します')
    logger.info('最大行数: {}'.format(data_row_max))
    logger.info('対象ファイル数: {}'.format(len(file_list)))
    logger.info('分割後のファイル数: {}'.format(file_count))

    # サブディレクトリの作成
    for MSM in MSM_list:
        taget_dir = os.path.join(temp_path, MSM) + os.sep
        logger.debug('作業ディレクトリの作成: {}'.format(taget_dir))
        os.makedirs(taget_dir, exist_ok=True)

    # 分割処理
    for i in range(file_count):
        logger.info('分割数{}/{}を処理しています'.format(i+1, file_count))

        # 読込
        index_start = i * data_row_max
        file_save_list = file_list[index_start:index_start + data_row_max]
        df_save = pd_read_pikle_map(file_save_list).sort_values('index')

        # 読み込んだファイルを削除
        for file in file_save_list:
            os.remove(file)
            logger.info('ファイル削除: {}'.format(file))

        # 分割
        df_group = df_save.groupby(['index'])
        for name, group in df_group:
            save_path = \
                os.path.join(temp_path, str(name), '{}_{}.pkl'.format(name, i))
            group.to_pickle(save_path)
            logger.info('分割ファイルの保存 {}'.format(save_path))


def combine_files(MSM_list, temp_path, path_save, logger):
    """ファイルの結合

    Args:
      MSM_list: MSMのリスト
      temp_path: 分割ファイルがあるディレクトリ
      path_save: 結合結果を保存するディレクトリ
      logger: ロガー

    Returns:

    """
    logger.info("分割したファイルを結合しています")

    for MSM in MSM_list:
        # 結合するMSMファイルがあるディレクトリ
        taget_dir = os.path.join(temp_path, MSM) + os.sep
        logger.info('結合ディレクトリ: {}'.format(taget_dir))

        # 結合するファイルを列挙
        file_list = sorted(glob.glob(os.path.join(taget_dir, '*.pkl')))

        # Pickelを取り出し結合されたデータフレームを作成
        df_save = pd_read_pikle_map(file_list) \
            .sort_values('date').set_index('date') \
            .drop('index', axis=1)

        # MSMへの保存(Gzip圧縮)
        MSM_gz_path = os.path.join(path_save, MSM + '.csv.gz')
        logger.info('MSMファイル保存: {}'.format(MSM_gz_path))
        df_save.to_csv(MSM_gz_path, compression='gzip')

        # 結合処理が終わったのでディレクトリを削除
        logger.info('ディレクトリ削除: {}'.format(taget_dir))
        shutil.rmtree(taget_dir)

    # 一時ディレクトリの削除
    shutil.rmtree(temp_path)

    logger.info("分割したファイルを結合しました")


if __name__ == '__main__':
    main()
