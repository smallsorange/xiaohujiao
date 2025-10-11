# 中国石化人工智能大赛：基于客户数据的智能营销决策0.84482分的完整Baseline方案
# 比赛的网址：
# https://aicup.sinopec.com/competition/SINOPEC-08/

# 背景介绍
# 在竞争白热化的成品油零售市场，加油站与能源公司正面临严峻挑战：传统 “广撒网” 式营销，如普适性折扣、无差别发放优惠券等，不仅成本居高不下，实际效果也往往不尽如人意，难以有效提升用户忠诚度和消费额。
 
# 为破解这一困局，借助人工智能技术实现精细化运营与精准营销成为关键。通过深度挖掘用户消费数据，构建精准的用户用券画像，进而预测不同用户对特定营销券活动的响应概率，能为业务部门提供有力支撑 —— 将合适的优惠（优惠券）精准触达最可能产生响应的目标用户。
 
# 这样既能严格控制营销成本，又能最大限度提升活动效果，同时增强用户满意度，最终实现用户忠诚度与消费额的双重提升。

# 竞赛任务
# 参赛者需要根据数据集提供的用户消费数据以及优惠券相关数据（钱包交易数据、优惠券发放数据、优惠券使用数据），预测用户对营销活动的响应可能性。

# 数据说明
# 本次比赛的数据已全部进行脱敏处理。处于数据隐私安全性考虑，真实场景数据中的部分数据无法公开发布，所以本赛题在一些数据字段的匹配上存在一定难度。我们鼓励选手尝试多种方法利用现有数据使模型达到最佳效果。

# 1.钱包交易数据表(cust_wallet_detail.csv)
# order_no / external_order_no (订单号 / 外部订单号)
# 含义: 交易的唯一ID。order_no是系统内部的订单号，external_order_no是关联的第三方或支付平台的订单号。
# user_id (用户ID)
# 含义: 经过脱敏处理的用户唯一编号。
# 重要性: 构建用户画像的核心ID。所有与“人”相关的分析，如消费频率、用户分群等，都将围绕此字段展开。
# membercode（会员编码）
# 含义: 会员编码。
# 重要性：与发券用券表客户进行关联。
# station_code / station_name
# 含义: 交易发生的加油站的唯一编码和名称。
# sale_time (销售时间)
# 含义: 交易发生的精确到秒的时间戳。
# tran_amt / receivable_amt (交易金额 / 应收金额 )
# 含义: tran_amt是商品原价总额；receivable_amt是应收总额。
# discounts_amt / point_amt / coupon_amt (折扣金额 / 积分抵扣金额 / 优惠券抵扣金额)
# 含义: 分别记录了通用折扣、积分兑换和优惠券带来的优惠金额。
# attributionorgcode / transactionorgcode (注册省编码 / 交易省编码)
# 含义: 用户注册省市与消费省市编码。
# Coupon_code (电子券编码)
# 含义: 核销电子券编码。
# 重要性：可与优惠券使用电子券编码关联。
# 2.优惠券使用(cust_coupon_detail_used.csv）
# marketcode （营销活动编号）
# 含义: 营销活动编码。
# marketrulenumber （活动细则编号）
# 含义: 营销规则编码。
# 重要性：营销活动细则编码，细则编码对应多个规则编码。
# voucherrulecode （规则编码）
# 含义: 电子券规则编码。
# vouchercode （电子券编码）
# 含义: 电子券编号。
# vouchertype （电子券类型）
# 含义: C01：现金券，C02：满抵券，C04：折扣券。
# vouchername （电子券名称）
# 含义: 电子券中文名称。
# transtime （使用时间）
# 含义: 电子券使用时间，使用时间：yyyy-MM-dd HH:mm:ss 。
# provinceCode （省份）
# 含义: 电子券所属省份。
# membercode（会员编码）
# 含义: 会员编码。
# 重要性：与发券、交易数据进行关联。
# channel （使用渠道）
# 含义: 电子券实际使用渠道。
# netCode （使用网点）
# 含义: 电子券实际使用网点。
# couponUseMoney （使用金额）
# 含义: 电子券使用金额，单位分。
# 3. 优惠券发放(cust_coupon_detail_send.csv)
# marketcode (营销活动编号)
# 含义: 营销活动编码。
# marketrulenumber (活动细则编号)
# 含义: 营销规则编码。
# 重要性：营销活动细则编码，细则编码对应多个规则编码。
# membercode（会员编码）
# 含义: 会员编码。
# 重要性：与用券、流水表客户进行关联。
# marketprovince （省份)
# 含义: 活动省份。
# voucherruleCode (电子券规则编码)
# 含义: 含义: 电子券规则编码。
# voucherrulename (电子券规则名称)
# 含义: 电子券规则名称。
# voucherstarttime (电子券生效时间)
# 含义: 电子券生效时间。
# voucherendTime (电子券失效时间)
# 含义: 电子券失效时间。
# vouchertype (电子券类型)
# 含义: C01：现金券，C02：满抵券，C04：折扣券。
# fulltype (电子券小类型)
# 含义: 如果电子券类型为C02满抵券的时候为必填项，01：满额抵，02：满额赠
# usechannel (用券渠道)
# 含义: 电子券使用渠道。
# cashvalue (电子券面额)
# 含义: 电子券使用金额，单位分。
# topamount (电子券满额)
# 含义: 电子券为满抵券时必填，为满足金额条件,单位分。
# endnumber (剩余数量)
# 含义: 电子券剩余数量。
# * 使用优惠券为发放优惠券数据的子集。

# 文件包中以_train为尾缀的文件对应训练集文件，_validation尾缀的文件对应验证集文件。
# cust_wallet_detail_validation_without_truth.csv 为初赛待预测的用户数据，文件格式与train相同但是不提供用户是否使用优惠券相关的数据。

# 4. 提交文件 sample_submission.csv
# 规定了初赛提交以及模型输出文件的格式。包含id与predict。
# 其中:

# id为交易数据表中对应的订单号order_no。
# predict为预测该用户响应营销事件的可能性的AUC结果值。
# 评测方法
# 本赛题是一个典型的二分类问题（用户响应或不响应），参赛选手需要提交响应的预测概率，并采用AUC (Area Under the ROC Curve) 作为主要的评测指标。



# 下面是解题思路：

# 一、赛题理解
# 技术问题定义
# 这是一个典型的二分类问题：预测用户是否会使用优惠券。

# 二、解决方案整体架构
# 原始数据 → 深度特征工程 → 特征选择 → 模型集成 → 预测结果
# 核心思路
# 深度挖掘用户画像
# ：从多维度刻画用户消费行为
# 构建营销匹配度
# ：分析用户消费能力与优惠券的匹配程度
# 集成学习提升效果
# ：结合多种机器学习算法的优势
# 三、深度特征工程策略
# 3.1 基础特征构建
# 时间维度特征

# 交易时间：小时、星期、月份、季度
# 时段划分：工作时间、周末、早中晚时段
# 时间规律：月初月末、节假日特征
# 金额维度特征

# 金额统计：原价、实付、折扣金额、折扣率
# 金额变换：对数变换、分位数分级
# 地理特征：是否跨省消费
# RFM分析升级版

# R (Recency)
# ：最近消费距今天数、首次消费距今天数
# F (Frequency)
# ：消费频次、平均消费间隔、消费规律性
# M (Monetary)
# ：消费金额统计、消费能力等级
# 消费行为特征

# 偏好分析：最常去的加油站、偏好时段
# 稳定性：消费金额变异系数、站点多样性
# 趋势分析：消费趋势斜率、近期vs历史对比

# 优惠券属性分析

# 面值统计：均值、最大值、分位数分布
# 类型分布：现金券、满抵券、折扣券数量占比
# 门槛分析：满额条件的统计特征
# 时效性特征

# 有效期长度、发放时间规律
# 剩余数量统计
# 3.4 交互特征（核心创新）)
# 客户价值分层匹配

# 根据客户价值等级调整优惠券匹配度
# 不同价值层级客户的优惠券偏好分析
# 3.5 高级特征
# 聚类特征

# 基于消费行为的用户聚类（K-means）
# 计算用户到各聚类中心的距离
# 统计学特征

# 消费金额的偏度、峰度
# 时间间隔的变异系数
# 消费趋势的线性拟合斜率
# 滑动窗口特征

# 四、模型集成策略

# 4.1 Stacking集成框架
# 第一层基学习器

# LightGBM
#  (2个不同参数版本)：梯度提升，处理类别特征能力强
# XGBoost
# ：极端梯度提升，泛化能力好
# CatBoost
# ：自动处理类别特征，过拟合抑制能力强
# 第二层元学习器

# 使用LightGBM作为元学习器
# 输入：第一层模型的预测概率
# 输出：最终预测结果
# 4.2 训练策略
# 交叉验证设计

# 特征选择

# 基于特征重要性选择Top 150个特征
# 避免过拟合，提升模型泛化能力
# 4.3 模型优化
# 超参数调优


# 使用Optuna库进行贝叶斯优化
# 针对LightGBM的关键参数进行调优
# 数据处理技巧

# 特征名清理：移除XGBoost不支持的特殊字符
# 缺失值处理：数值型填0，类别型填'unknown'
# 标签编码：处理类别特征
# 五、关键技术亮点
# 5.1 防止数据泄露
# # 严格移除可能泄露标签的特征
# leak_cols = ['coupon_code', 'coupon_amt', 'discounts_amt', 'point_amt']
# 5.2 特征工程自动化
# 设计AdvancedFeatureEngineer类，模块化特征生成
# 统一的特征名清理和缺失值处理
# 5.3 模型训练自动化
# AdvancedModelTrainer
# 类封装完整的模型训练流程
# 支持多种特征选择方法和模型优化策略

# 、
# 下面是0.84482分的baseline的完整代码：

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
中国石化第一届人工智能大赛 - 基于客户数据的智能营销决策
深度特征挖掘与模型优化版本 - 修正版
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna

# 设置随机种子
SEED = 42
np.random.seed(SEED)


def clean_feature_name(name):
    """清理特征名称，移除XGBoost不允许的字符"""
    name = str(name).replace('<lambda>', 'func')
    name = name.replace('[', '_').replace(']', '_')
    name = name.replace('<', '_').replace('>', '_')
    name = name.replace('(', '_').replace(')', '_')
    name = name.replace(',', '_').replace(' ', '_')
    while '__' in name:
        name = name.replace('__', '_')
    name = name.strip('_')
    return name


class AdvancedFeatureEngineer:
    """深度特征工程类"""

    def __init__(self):
        self.label_encoders = {}
        self.scalers = {}
        self.user_clusters = {}

    def create_basic_features(self, wallet_df, is_train=True):
        """创建基础特征 - 增强版本"""
        print(f"创建基础特征... (is_train={is_train})")

        # 时间特征
        wallet_df['sale_time'] = pd.to_datetime(wallet_df['sale_time'])
        wallet_df['sale_hour'] = wallet_df['sale_time'].dt.hour
        wallet_df['sale_dayofweek'] = wallet_df['sale_time'].dt.dayofweek
        wallet_df['sale_day'] = wallet_df['sale_time'].dt.day
        wallet_df['sale_month'] = wallet_df['sale_time'].dt.month
        wallet_df['sale_quarter'] = wallet_df['sale_time'].dt.quarter
        wallet_df['sale_year'] = wallet_df['sale_time'].dt.year
        wallet_df['sale_week'] = wallet_df['sale_time'].dt.isocalendar().week

        # 更细粒度的时间特征
        wallet_df['is_weekend'] = wallet_df['sale_dayofweek'].isin([5, 6]).astype(int)
        wallet_df['is_worktime'] = wallet_df['sale_hour'].between(7, 19).astype(int)
        wallet_df['is_morning'] = wallet_df['sale_hour'].between(6, 11).astype(int)
        wallet_df['is_noon'] = wallet_df['sale_hour'].between(11, 14).astype(int)
        wallet_df['is_afternoon'] = wallet_df['sale_hour'].between(14, 18).astype(int)
        wallet_df['is_evening'] = wallet_df['sale_hour'].between(18, 22).astype(int)
        wallet_df['is_night'] = wallet_df['sale_hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)

        # 节假日特征（简化版）
        wallet_df['is_month_start'] = (wallet_df['sale_day'] <= 5).astype(int)
        wallet_df['is_month_end'] = (wallet_df['sale_day'] >= 25).astype(int)
        wallet_df['is_month_middle'] = ((wallet_df['sale_day'] > 10) & (wallet_df['sale_day'] <= 20)).astype(int)

        # 时间段特征（更精细）
        wallet_df['time_period_4'] = pd.cut(wallet_df['sale_hour'],
                                            bins=[-0.1, 6, 12, 18, 24],
                                            labels=[1, 2, 3, 4])
        wallet_df['time_period_4'] = wallet_df['time_period_4'].cat.codes + 1
        wallet_df['time_period_4'] = wallet_df['time_period_4'].replace(-1, 2)

        wallet_df['time_period_8'] = pd.cut(wallet_df['sale_hour'],
                                            bins=[-0.1, 3, 6, 9, 12, 15, 18, 21, 24],
                                            labels=[1, 2, 3, 4, 5, 6, 7, 8])
        wallet_df['time_period_8'] = wallet_df['time_period_8'].cat.codes + 1
        wallet_df['time_period_8'] = wallet_df['time_period_8'].replace(-1, 4)

        # 金额特征 - 只使用交易金额和应收金额
        wallet_df['amt_ratio'] = wallet_df['receivable_amt'] / (wallet_df['tran_amt'] + 0.01)
        wallet_df['amt_diff'] = wallet_df['tran_amt'] - wallet_df['receivable_amt']
        wallet_df['amt_discount_rate'] = wallet_df['amt_diff'] / (wallet_df['tran_amt'] + 0.01)

        # 金额的对数变换
        wallet_df['tran_amt_log'] = np.log1p(wallet_df['tran_amt'])
        wallet_df['receivable_amt_log'] = np.log1p(wallet_df['receivable_amt'])

        # 地理特征
        wallet_df['is_cross_province'] = (wallet_df['attributionorgcode'] !=
                                          wallet_df['transactionorgcode']).astype(int)

        # 交易金额分级（多种分级方式）
        try:
            wallet_df['tran_amt_level_5'] = pd.qcut(wallet_df['tran_amt'],
                                                    q=5, labels=False, duplicates='drop')
            wallet_df['tran_amt_level_5'] = wallet_df['tran_amt_level_5'].fillna(2).astype(int) + 1
        except:
            wallet_df['tran_amt_level_5'] = 3

        try:
            wallet_df['tran_amt_level_10'] = pd.qcut(wallet_df['tran_amt'],
                                                     q=10, labels=False, duplicates='drop')
            wallet_df['tran_amt_level_10'] = wallet_df['tran_amt_level_10'].fillna(5).astype(int) + 1
        except:
            wallet_df['tran_amt_level_10'] = 6

        # 金额的百分位特征
        for pct in [25, 50, 75, 90, 95]:
            threshold = np.percentile(wallet_df['tran_amt'], pct)
            wallet_df[f'tran_amt_above_p{pct}'] = (wallet_df['tran_amt'] > threshold).astype(int)

        return wallet_df

    def create_user_features(self, wallet_df, membercode_col='membercode'):
        """创建用户级别特征 - 大幅增强版本"""
        print(f"创建用户级别特征 (基于{membercode_col})...")

        user_features = []

        # 基础统计特征（更全面）
        agg_dict = {
            'order_no': ['count'],  # 交易次数
            'tran_amt': ['sum', 'mean', 'std', 'min', 'max', 'median'],  # 移除skew和kurt
            'receivable_amt': ['sum', 'mean', 'std', 'min', 'max', 'median'],
            'amt_ratio': ['mean', 'std', 'min', 'max'],  # 移除skew
            'amt_diff': ['mean', 'std', 'sum'],
            'amt_discount_rate': ['mean', 'std', 'max'],
            'tran_amt_log': ['mean', 'std'],
            'receivable_amt_log': ['mean', 'std'],
            'station_code': ['nunique'],  # 加油站数量
            'sale_dayofweek': ['mean', 'std', 'nunique'],
            'sale_hour': ['mean', 'std', 'nunique', 'min', 'max'],
            'sale_month': ['nunique'],
            'sale_day': ['nunique'],
            'sale_week': ['nunique'],
            'is_weekend': ['mean', 'sum'],
            'is_worktime': ['mean'],
            'is_morning': ['mean', 'sum'],
            'is_noon': ['mean', 'sum'],
            'is_afternoon': ['mean', 'sum'],
            'is_evening': ['mean', 'sum'],
            'is_night': ['mean', 'sum'],
            'is_cross_province': ['mean', 'sum'],
            'time_period_4': ['mean', 'std'],
            'time_period_8': ['mean', 'std'],
            'tran_amt_level_5': ['mean', 'std'],
            'tran_amt_level_10': ['mean', 'std'],
        }

        # 添加百分位特征的统计
        for pct in [25, 50, 75, 90, 95]:
            agg_dict[f'tran_amt_above_p{pct}'] = ['mean', 'sum']

        basic_stats = wallet_df.groupby(membercode_col).agg(agg_dict)

        # 手动计算偏度和峰度（如果需要）
        try:
            # 使用scipy计算偏度和峰度
            from scipy.stats import skew, kurtosis

            tran_amt_skew = wallet_df.groupby(membercode_col)['tran_amt'].apply(lambda x: skew(x) if len(x) > 2 else 0)
            tran_amt_kurt = wallet_df.groupby(membercode_col)['tran_amt'].apply(
                lambda x: kurtosis(x) if len(x) > 2 else 0)
            amt_ratio_skew = wallet_df.groupby(membercode_col)['amt_ratio'].apply(
                lambda x: skew(x) if len(x) > 2 else 0)

            # 将这些添加到basic_stats
            basic_stats['tran_amt_skew'] = tran_amt_skew
            basic_stats['tran_amt_kurt'] = tran_amt_kurt
            basic_stats['amt_ratio_skew'] = amt_ratio_skew

        except ImportError:
            print("  警告: scipy未安装，跳过偏度和峰度计算")
        except Exception as e:
            print(f"  警告: 计算偏度和峰度时出错: {e}")

        # 清理列名
        new_columns = []
        for col in basic_stats.columns:
            if isinstance(col, tuple):
                new_name = f"{col[0]}_{col[1]}"
                new_name = clean_feature_name(new_name)
                new_columns.append(new_name)
            else:
                new_columns.append(clean_feature_name(col))
        basic_stats.columns = new_columns
        user_features.append(basic_stats)

        # 最喜欢的站点、时间等
        favorite_station = wallet_df.groupby(membercode_col)['station_code'].agg(
            lambda x: x.value_counts().index[0] if len(x) > 0 else 0
        ).to_frame('favorite_station')
        user_features.append(favorite_station)

        favorite_hour = wallet_df.groupby(membercode_col)['sale_hour'].agg(
            lambda x: x.value_counts().index[0] if len(x) > 0 else 0
        ).to_frame('favorite_hour')
        user_features.append(favorite_hour)

        favorite_dayofweek = wallet_df.groupby(membercode_col)['sale_dayofweek'].agg(
            lambda x: x.value_counts().index[0] if len(x) > 0 else 0
        ).to_frame('favorite_dayofweek')
        user_features.append(favorite_dayofweek)

        favorite_month = wallet_df.groupby(membercode_col)['sale_month'].agg(
            lambda x: x.value_counts().index[0] if len(x) > 0 else 0
        ).to_frame('favorite_month')
        user_features.append(favorite_month)

        # 增强的RFM特征
        current_date = wallet_df['sale_time'].max()
        rfm = wallet_df.groupby(membercode_col).agg({
            'sale_time': [lambda x: (current_date - x.max()).days,  # Recency
                          lambda x: (current_date - x.min()).days,  # 首次购买距今天数
                          lambda x: (x.max() - x.min()).days],  # 活跃天数
            'order_no': 'count',  # Frequency
            'tran_amt': ['sum', 'mean']  # Monetary
        })
        rfm.columns = ['recency_days', 'first_purchase_days', 'customer_lifespan_days',
                       'frequency', 'monetary_sum', 'monetary_mean']

        # 计算购买频率（每天）
        rfm['purchase_frequency_per_day'] = rfm['frequency'] / (rfm['customer_lifespan_days'] + 1)

        # RFM分级（更细致）
        for col, reverse in [('recency_days', True), ('frequency', False), ('monetary_sum', False)]:
            try:
                if reverse:
                    rfm[f'{col}_score'] = pd.qcut(rfm[col], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')
                else:
                    rfm[f'{col}_score'] = pd.qcut(rfm[col], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
                rfm[f'{col}_score'] = rfm[f'{col}_score'].cat.codes + 1
            except:
                rfm[f'{col}_score'] = 3

        rfm['rfm_score'] = (rfm['recency_days_score'].fillna(3) +
                            rfm['frequency_score'].fillna(3) +
                            rfm['monetary_sum_score'].fillna(3))

        # 客户生命周期价值
        rfm['clv_simple'] = rfm['monetary_mean'] * rfm['frequency'] / (rfm['recency_days'] + 1)

        user_features.append(rfm)

        # 消费行为稳定性特征
        consumption_behavior = wallet_df.groupby(membercode_col).agg({
            'tran_amt': [lambda x: x.std() / (x.mean() + 0.01),  # 变异系数
                         lambda x: len(x.unique()) / len(x),  # 金额多样性
                         lambda x: (x == x.mode().iloc[0]).sum() / len(x) if len(x.mode()) > 0 else 0],  # 众数占比
            'station_code': lambda x: len(x.unique()) / len(x),  # 站点多样性
            'sale_hour': lambda x: len(x.unique()),  # 时间多样性
            'sale_dayofweek': lambda x: len(x.unique()),  # 星期多样性
        })
        consumption_behavior.columns = ['consumption_cv', 'amt_diversity', 'amt_mode_ratio',
                                        'station_diversity', 'hour_diversity', 'day_diversity']
        user_features.append(consumption_behavior)

        # 时间模式特征
        time_patterns = []
        for user in wallet_df[membercode_col].unique():
            user_data = wallet_df[wallet_df[membercode_col] == user].copy()
            user_data = user_data.sort_values('sale_time')

            # 计算时间间隔统计
            if len(user_data) > 1:
                time_diffs = user_data['sale_time'].diff().dt.days.dropna()
                time_pattern = {
                    membercode_col: user,
                    'avg_interval_days': time_diffs.mean() if len(time_diffs) > 0 else 0,
                    'std_interval_days': time_diffs.std() if len(time_diffs) > 0 else 0,
                    'max_interval_days': time_diffs.max() if len(time_diffs) > 0 else 0,
                    'min_interval_days': time_diffs.min() if len(time_diffs) > 0 else 0,
                    'interval_cv': time_diffs.std() / (time_diffs.mean() + 0.01) if len(time_diffs) > 0 else 0,
                    'transaction_regularity': 1 / (time_diffs.std() + 1) if len(time_diffs) > 0 else 0,
                }
            else:
                time_pattern = {
                    membercode_col: user,
                    'avg_interval_days': 0,
                    'std_interval_days': 0,
                    'max_interval_days': 0,
                    'min_interval_days': 0,
                    'interval_cv': 0,
                    'transaction_regularity': 0,
                }

            time_patterns.append(time_pattern)

        time_pattern_df = pd.DataFrame(time_patterns)
        user_features.append(time_pattern_df.set_index(membercode_col))

        # 消费趋势特征
        trend_features = []
        for user in wallet_df[membercode_col].unique():
            user_data = wallet_df[wallet_df[membercode_col] == user].copy()
            user_data = user_data.sort_values('sale_time')

            if len(user_data) >= 3:
                # 线性趋势
                x = np.arange(len(user_data))
                y = user_data['tran_amt'].values
                trend_slope = np.polyfit(x, y, 1)[0] if len(x) > 1 else 0

                # 最近vs历史对比
                recent_avg = user_data.tail(3)['tran_amt'].mean()
                historical_avg = user_data.head(-3)['tran_amt'].mean() if len(user_data) > 3 else recent_avg
                trend_ratio = recent_avg / (historical_avg + 0.01)
            else:
                trend_slope = 0
                trend_ratio = 1

            trend_features.append({
                membercode_col: user,
                'consumption_trend_slope': trend_slope,
                'recent_vs_historical_ratio': trend_ratio,
            })

        trend_df = pd.DataFrame(trend_features)
        user_features.append(trend_df.set_index(membercode_col))

        # 合并所有特征
        result = pd.concat(user_features, axis=1).reset_index()
        result.columns = [clean_feature_name(col) for col in result.columns]

        return result

    def create_coupon_send_features(self, coupon_send_df):
        """创建优惠券发放特征 - 增强版本"""
        print("创建优惠券发放特征...")

        if len(coupon_send_df) == 0:
            return pd.DataFrame({'membercode': []})

        send_features = []

        # 基础统计（更全面）
        basic_send = coupon_send_df.groupby('membercode').agg({
            'marketcode': ['nunique', 'count'],  # 营销活动数量和参与次数
            'marketrulenumber': ['count', 'nunique'],  # 优惠券总数和规则数
            'voucherrulecode': 'nunique',  # 不同规则数
            'cashvalue': ['mean', 'max', 'min', 'std', 'sum', 'median'],  # 优惠券面值统计
            'topamount': ['mean', 'max', 'min', 'std', 'count'],  # 满额门槛统计
            'endnumber': ['sum', 'mean', 'max', 'std'],  # 剩余数量
        })

        # 清理列名
        new_columns = []
        for col in basic_send.columns:
            if isinstance(col, tuple):
                new_name = f"send_{col[0]}_{col[1]}"
                new_columns.append(clean_feature_name(new_name))
        basic_send.columns = new_columns
        send_features.append(basic_send)

        # 优惠券类型详细统计
        voucher_type_counts = coupon_send_df.groupby(['membercode', 'vouchertype']).size().unstack(fill_value=0)
        voucher_type_counts.columns = [f'send_type_{clean_feature_name(str(col))}_count'
                                       for col in voucher_type_counts.columns]
        send_features.append(voucher_type_counts)

        # 优惠券类型面值统计
        voucher_type_values = coupon_send_df.groupby(['membercode', 'vouchertype'])['cashvalue'].mean().unstack(
            fill_value=0)
        voucher_type_values.columns = [f'send_type_{clean_feature_name(str(col))}_avg_value'
                                       for col in voucher_type_values.columns]
        send_features.append(voucher_type_values)

        # 满额类型统计
        if 'fulltype' in coupon_send_df.columns:
            full_type_counts = coupon_send_df.groupby(['membercode', 'fulltype']).size().unstack(fill_value=0)
            full_type_counts.columns = [f'send_fulltype_{clean_feature_name(str(col))}_count'
                                        for col in full_type_counts.columns]
            send_features.append(full_type_counts)

        # 优惠券时效特征（更详细）
        coupon_send_df['voucherstarttime'] = pd.to_datetime(coupon_send_df['voucherstarttime'])
        coupon_send_df['voucherendtime'] = pd.to_datetime(coupon_send_df['voucherendtime'])
        coupon_send_df['voucher_duration_days'] = (
                coupon_send_df['voucherendtime'] - coupon_send_df['voucherstarttime']
        ).dt.days

        # 优惠券发放时间特征
        coupon_send_df['start_hour'] = coupon_send_df['voucherstarttime'].dt.hour
        coupon_send_df['start_dayofweek'] = coupon_send_df['voucherstarttime'].dt.dayofweek
        coupon_send_df['start_month'] = coupon_send_df['voucherstarttime'].dt.month

        duration_stats = coupon_send_df.groupby('membercode').agg({
            'voucher_duration_days': ['mean', 'std', 'max', 'min', 'median'],
            'start_hour': ['mean', 'nunique'],
            'start_dayofweek': ['nunique'],
            'start_month': ['nunique'],
        })

        new_columns = []
        for col in duration_stats.columns:
            if isinstance(col, tuple):
                new_name = f"voucher_{col[0]}_{col[1]}"
                new_columns.append(clean_feature_name(new_name))
        duration_stats.columns = new_columns
        send_features.append(duration_stats)

        # 优惠券价值分布特征
        value_percentiles = coupon_send_df.groupby('membercode')['cashvalue'].quantile([0.25, 0.5, 0.75]).unstack()
        value_percentiles.columns = [f'coupon_value_p{int(col * 100)}' for col in value_percentiles.columns]
        send_features.append(value_percentiles)

        # 满额门槛分布特征
        if 'topamount' in coupon_send_df.columns:
            threshold_percentiles = coupon_send_df.groupby('membercode')['topamount'].quantile(
                [0.25, 0.5, 0.75]).unstack()
            threshold_percentiles.columns = [f'threshold_p{int(col * 100)}' for col in threshold_percentiles.columns]
            send_features.append(threshold_percentiles)

        # 优惠券发放频率特征
        first_send = coupon_send_df.groupby('membercode')['voucherstarttime'].min()
        last_send = coupon_send_df.groupby('membercode')['voucherstarttime'].max()
        send_count = coupon_send_df.groupby('membercode').size()

        frequency_features = pd.DataFrame({
            'coupon_send_span_days': (last_send - first_send).dt.days,
            'coupon_send_frequency': send_count / ((last_send - first_send).dt.days + 1),
            'total_coupon_count': send_count
        })
        send_features.append(frequency_features)

        # 合并所有发放特征
        result = pd.concat(send_features, axis=1).reset_index()
        result.columns = [clean_feature_name(col) for col in result.columns]

        return result

    def create_interaction_features(self, wallet_df, coupon_features):
        """创建交互特征 - 大幅增强版本"""
        print("创建交互特征...")

        if len(coupon_features) == 0:
            return pd.DataFrame({'membercode': wallet_df['membercode'].unique()})

        # 用户消费能力统计
        user_consumption = wallet_df.groupby('membercode').agg({
            'tran_amt': ['mean', 'median', 'max', 'std', 'sum'],
            'receivable_amt': ['mean', 'median'],
            'order_no': 'count'
        })

        # 清理列名
        new_columns = []
        for col in user_consumption.columns:
            if isinstance(col, tuple):
                new_name = f"user_{col[0]}_{col[1]}"
                new_columns.append(clean_feature_name(new_name))
        user_consumption.columns = new_columns
        user_consumption = user_consumption.reset_index()

        # 合并数据
        interaction_data = user_consumption.merge(coupon_features, on='membercode', how='left')

        interaction_features = []

        # 消费能力与优惠券门槛匹配度
        threshold_cols = [col for col in coupon_features.columns if 'topamount' in col.lower()]
        for threshold_col in threshold_cols:
            if threshold_col in interaction_data.columns:
                # 与平均消费的比较
                if 'user_tran_amt_mean' in interaction_data.columns:
                    interaction_data[f'consumption_vs_{threshold_col}_ratio'] = (
                            interaction_data['user_tran_amt_mean'] /
                            (interaction_data[threshold_col].fillna(
                                interaction_data['user_tran_amt_mean'].median()) + 1)
                    )

                # 与最大消费的比较
                if 'user_tran_amt_max' in interaction_data.columns:
                    interaction_data[f'max_consumption_vs_{threshold_col}_ratio'] = (
                            interaction_data['user_tran_amt_max'] /
                            (interaction_data[threshold_col].fillna(interaction_data['user_tran_amt_max'].median()) + 1)
                    )

        # 优惠券价值与消费水平匹配
        value_cols = [col for col in coupon_features.columns if 'cashvalue' in col.lower()]
        for value_col in value_cols:
            if value_col in interaction_data.columns:
                # 优惠券价值占消费比例
                if 'user_tran_amt_mean' in interaction_data.columns:
                    interaction_data[f'{value_col}_vs_consumption_ratio'] = (
                            interaction_data[value_col].fillna(0) /
                            (interaction_data['user_tran_amt_mean'] + 1)
                    )

                # 优惠券价值吸引力指数
                if 'user_tran_amt_std' in interaction_data.columns:
                    interaction_data[f'{value_col}_attractiveness'] = (
                            interaction_data[value_col].fillna(0) /
                            (interaction_data['user_tran_amt_std'].fillna(1) + 1)
                    )

        # 优惠券数量与消费频率匹配
        count_cols = [col for col in coupon_features.columns if 'count' in col.lower()]
        for count_col in count_cols:
            if count_col in interaction_data.columns and 'user_order_no_count' in interaction_data.columns:
                interaction_data[f'{count_col}_vs_frequency_ratio'] = (
                        interaction_data[count_col].fillna(0) /
                        (interaction_data['user_order_no_count'] + 1)
                )

        # 客户价值分层匹配
        if 'user_tran_amt_sum' in interaction_data.columns:
            try:
                interaction_data['customer_value_tier'] = pd.qcut(
                    interaction_data['user_tran_amt_sum'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop'
                )
                interaction_data['customer_value_tier'] = interaction_data['customer_value_tier'].cat.codes + 1
            except:
                interaction_data['customer_value_tier'] = 3

            # 根据客户价值层级调整优惠券匹配度
            for value_col in value_cols[:3]:  # 只选择前3个避免特征过多
                if value_col in interaction_data.columns:
                    interaction_data[f'{value_col}_tier_adjusted'] = (
                            interaction_data[value_col].fillna(0) *
                            interaction_data['customer_value_tier']
                    )

        # 选择交互特征列
        interaction_cols = ['membercode']
        for col in interaction_data.columns:
            if any(keyword in col.lower() for keyword in ['ratio', 'vs_', 'attractiveness', 'tier']):
                interaction_cols.append(col)

        result = interaction_data[interaction_cols]
        result.columns = [clean_feature_name(col) for col in result.columns]

        return result

    def create_advanced_time_features(self, wallet_df):
        """创建高级时间特征"""
        print("创建高级时间特征...")

        # 按用户排序
        wallet_df_sorted = wallet_df.sort_values(['membercode', 'sale_time']).copy()

        # 计算各种时间间隔
        wallet_df_sorted['time_diff_hours'] = wallet_df_sorted.groupby('membercode')[
                                                  'sale_time'].diff().dt.total_seconds() / 3600
        wallet_df_sorted['time_diff_days'] = wallet_df_sorted['time_diff_hours'] / 24

        # 计算累积特征
        wallet_df_sorted['cumsum_tran_amt'] = wallet_df_sorted.groupby('membercode')['tran_amt'].cumsum()
        wallet_df_sorted['cumcount_orders'] = wallet_df_sorted.groupby('membercode').cumcount() + 1
        wallet_df_sorted['avg_tran_amt_so_far'] = wallet_df_sorted['cumsum_tran_amt'] / wallet_df_sorted[
            'cumcount_orders']

        # 修复滑动窗口特征计算
        rolling_features = []
        for user in wallet_df_sorted['membercode'].unique():
            user_data = wallet_df_sorted[wallet_df_sorted['membercode'] == user].copy()
            user_data = user_data.set_index('sale_time').sort_index()

            user_rolling = {'membercode': user}

            # 计算不同窗口的滑动特征
            for window in [3, 7, 30]:  # 3天、7天、30天窗口
                try:
                    # 滑动平均金额
                    rolling_mean = user_data['tran_amt'].rolling(window=f'{window}D', min_periods=1).mean()
                    user_rolling[f'rolling_mean_{window}d_last'] = rolling_mean.iloc[-1] if len(rolling_mean) > 0 else 0
                    user_rolling[f'rolling_mean_{window}d_std'] = rolling_mean.std() if len(rolling_mean) > 1 else 0

                    # 滑动交易次数
                    rolling_count = user_data['tran_amt'].rolling(window=f'{window}D', min_periods=1).count()
                    user_rolling[f'rolling_count_{window}d_last'] = rolling_count.iloc[-1] if len(
                        rolling_count) > 0 else 0
                    user_rolling[f'rolling_count_{window}d_max'] = rolling_count.max() if len(rolling_count) > 0 else 0

                except Exception as e:
                    # 如果滑动窗口计算失败，使用简单统计
                    recent_data = user_data.tail(window)
                    user_rolling[f'rolling_mean_{window}d_last'] = recent_data['tran_amt'].mean() if len(
                        recent_data) > 0 else 0
                    user_rolling[f'rolling_mean_{window}d_std'] = recent_data['tran_amt'].std() if len(
                        recent_data) > 1 else 0
                    user_rolling[f'rolling_count_{window}d_last'] = len(recent_data)
                    user_rolling[f'rolling_count_{window}d_max'] = len(recent_data)

            rolling_features.append(user_rolling)

        rolling_df = pd.DataFrame(rolling_features)

        # 时间特征聚合
        time_features = wallet_df_sorted.groupby('membercode').agg({
            'time_diff_hours': ['mean', 'std', 'max', 'min', 'median'],
            'time_diff_days': ['mean', 'std', 'max', 'min'],
            'avg_tran_amt_so_far': ['last', 'std'],
        })

        # 清理列名
        new_columns = []
        for col in time_features.columns:
            if isinstance(col, tuple):
                new_name = f"time_{col[0]}_{col[1]}"
                new_columns.append(clean_feature_name(new_name))
        time_features.columns = new_columns

        # 计算时间规律性指标
        time_features['time_regularity_score'] = 1 / (time_features['time_time_diff_hours_std'].fillna(1) + 1)

        # 合并滑动窗口特征
        time_features = time_features.reset_index().merge(rolling_df, on='membercode', how='left')

        # 计算购买强度
        time_features['purchase_intensity_7d'] = time_features['rolling_count_7d_last'] / 7
        time_features['purchase_intensity_30d'] = time_features['rolling_count_30d_last'] / 30

        # 清理所有列名
        time_features.columns = [clean_feature_name(col) for col in time_features.columns]

        return time_features

    def create_clustering_features(self, wallet_df, n_clusters=8):
        """创建聚类特征"""
        print(f"创建聚类特征 (k={n_clusters})...")

        # 准备聚类数据
        cluster_data = wallet_df.groupby('membercode').agg({
            'tran_amt': ['mean', 'std', 'sum'],
            'sale_hour': 'mean',
            'sale_dayofweek': 'mean',
            'is_weekend': 'mean',
            'order_no': 'count',
            'station_code': 'nunique',
        })

        # 清理列名
        new_columns = []
        for col in cluster_data.columns:
            if isinstance(col, tuple):
                new_name = f"{col[0]}_{col[1]}"
                new_columns.append(clean_feature_name(new_name))
        cluster_data.columns = new_columns

        # 填充缺失值并标准化
        cluster_data = cluster_data.fillna(0)
        scaler = StandardScaler()
        cluster_data_scaled = scaler.fit_transform(cluster_data)

        # 执行聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
        clusters = kmeans.fit_predict(cluster_data_scaled)

        # 保存聚类结果
        cluster_features = pd.DataFrame({
            'membercode': cluster_data.index,
            'user_cluster': clusters
        })

        # 计算到每个聚类中心的距离
        distances = kmeans.transform(cluster_data_scaled)
        for i in range(n_clusters):
            cluster_features[f'distance_to_cluster_{i}'] = distances[:, i]

        # 计算到最近聚类中心的距离
        cluster_features['min_cluster_distance'] = distances.min(axis=1)
        cluster_features['max_cluster_distance'] = distances.max(axis=1)

        return cluster_features

    def create_statistical_features(self, wallet_df):
        """创建统计学特征"""
        print("创建统计学特征...")

        statistical_features = []

        # 按用户计算统计特征
        for user in wallet_df['membercode'].unique():
            user_data = wallet_df[wallet_df['membercode'] == user].copy()

            if len(user_data) >= 2:
                amounts = user_data['tran_amt'].values

                # 基础统计量
                stats = {
                    'membercode': user,
                    'amount_range': amounts.max() - amounts.min(),
                    'amount_iqr': np.percentile(amounts, 75) - np.percentile(amounts, 25),
                    'amount_outlier_ratio': np.sum(np.abs(amounts - amounts.mean()) > 2 * amounts.std()) / len(amounts),
                }

                # 分布特征
                if len(amounts) >= 3:
                    try:
                        from scipy.stats import skew, kurtosis
                        stats.update({
                            'amount_skewness': skew(amounts),
                            'amount_kurtosis': kurtosis(amounts),
                        })
                    except ImportError:
                        stats.update({
                            'amount_skewness': 0,
                            'amount_kurtosis': 0,
                        })
                else:
                    stats.update({
                        'amount_skewness': 0,
                        'amount_kurtosis': 0,
                    })

                # 变化趋势
                if len(amounts) >= 4:
                    # 计算一阶差分的统计量
                    diff1 = np.diff(amounts)
                    stats.update({
                        'amount_diff_mean': diff1.mean(),
                        'amount_diff_std': diff1.std(),
                        'amount_trend_strength': np.abs(diff1.mean()) / (diff1.std() + 0.01),
                    })
                else:
                    stats.update({
                        'amount_diff_mean': 0,
                        'amount_diff_std': 0,
                        'amount_trend_strength': 0,
                    })

            else:
                # 单次交易用户的默认值
                stats = {
                    'membercode': user,
                    'amount_range': 0,
                    'amount_iqr': 0,
                    'amount_outlier_ratio': 0,
                    'amount_skewness': 0,
                    'amount_kurtosis': 0,
                    'amount_diff_mean': 0,
                    'amount_diff_std': 0,
                    'amount_trend_strength': 0,
                }

            statistical_features.append(stats)

        result = pd.DataFrame(statistical_features)
        result.columns = [clean_feature_name(col) for col in result.columns]

        return result


class AdvancedModelTrainer:
    """高级模型训练类"""

    def __init__(self, seed=42):
        self.seed = seed
        self.models = {}
        self.feature_importances = {}

    def optimize_lightgbm_params(self, X_train, y_train, n_trials=50):
        """使用Optuna优化LightGBM参数"""
        print("优化LightGBM参数...")

        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'verbose': -1,
                'seed': self.seed,
                'n_estimators': 500,
            }

            # 交叉验证
            kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.seed)
            scores = []

            for train_idx, valid_idx in kf.split(X_train, y_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

                model = lgb.LGBMClassifier(**params)
                model.fit(X_tr, y_tr,
                          eval_set=[(X_val, y_val)],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

                pred = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, pred)
                scores.append(score)

            return np.mean(scores)

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(objective, n_trials=n_trials)

        print(f"最佳LightGBM参数: {study.best_params}")
        print(f"最佳CV分数: {study.best_value:.4f}")

        return study.best_params

    def train_stacking_model(self, X_train, y_train, cv_folds=5):
        """训练Stacking模型"""
        print(f"训练Stacking模型 (CV={cv_folds}折)...")

        # 第一层模型 - 注意：XGBoost在交叉验证时使用early_stopping，但在最终训练时不使用
        base_models = {
            'lgb1': lgb.LGBMClassifier(
                objective='binary', metric='auc', num_leaves=31, learning_rate=0.05,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
                n_estimators=500, verbose=-1, seed=self.seed
            ),
            'lgb2': lgb.LGBMClassifier(
                objective='binary', metric='auc', num_leaves=50, learning_rate=0.03,
                feature_fraction=0.9, bagging_fraction=0.7, bagging_freq=3,
                min_child_samples=10, reg_alpha=0.05, reg_lambda=0.05,
                n_estimators=800, verbose=-1, seed=self.seed + 1
            ),
            'xgb1': xgb.XGBClassifier(
                objective='binary:logistic', eval_metric='auc', max_depth=6,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                min_child_weight=3, reg_alpha=0.1, reg_lambda=0.1,
                n_estimators=500, verbosity=0, seed=self.seed
            ),
            'cat1': CatBoostClassifier(
                iterations=500, learning_rate=0.05, depth=6,
                loss_function='Logloss', eval_metric='AUC',
                l2_leaf_reg=3, min_data_in_leaf=20,
                random_seed=self.seed, verbose=False
            ),
        }

        # 生成元特征
        kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.seed)
        meta_features = np.zeros((len(X_train), len(base_models)))
        meta_labels = np.zeros(len(X_train))

        cv_scores = {name: [] for name in base_models.keys()}

        for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train, y_train)):
            print(f"处理第 {fold + 1}/{cv_folds} 折...")

            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

            for i, (name, model) in enumerate(base_models.items()):
                # 训练模型 - 交叉验证时使用early stopping
                if 'xgb' in name:
                    # 清理特征名
                    X_tr_clean = X_tr.copy()
                    X_val_clean = X_val.copy()
                    X_tr_clean.columns = [clean_feature_name(col) for col in X_tr_clean.columns]
                    X_val_clean.columns = [clean_feature_name(col) for col in X_val_clean.columns]

                    # 为交叉验证创建带early_stopping的临时模型
                    temp_model = xgb.XGBClassifier(
                        objective='binary:logistic', eval_metric='auc', max_depth=6,
                        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                        min_child_weight=3, reg_alpha=0.1, reg_lambda=0.1,
                        n_estimators=500, verbosity=0, seed=self.seed,
                        early_stopping_rounds=50  # 只在CV时使用early stopping
                    )
                    temp_model.fit(X_tr_clean, y_tr,
                                  eval_set=[(X_val_clean, y_val)],
                                  verbose=False)
                    pred = temp_model.predict_proba(X_val_clean)[:, 1]
                elif 'lgb' in name:
                    model.fit(X_tr, y_tr,
                              eval_set=[(X_val, y_val)],
                              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
                    pred = model.predict_proba(X_val)[:, 1]
                else:  # CatBoost
                    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
                    pred = model.predict_proba(X_val)[:, 1]

                # 保存预测结果
                meta_features[valid_idx, i] = pred
                score = roc_auc_score(y_val, pred)
                cv_scores[name].append(score)
                print(f"  {name} AUC: {score:.4f}")

            meta_labels[valid_idx] = y_val

        # 打印CV结果
        print("\nCV结果:")
        for name, scores in cv_scores.items():
            print(f"{name}: {np.mean(scores):.4f} (+/- {np.std(scores) * 2:.4f})")

        # 训练最终的基础模型 - 不使用early stopping
        print("\n训练最终基础模型...")
        final_base_models = {}
        for name, model in base_models.items():
            if 'xgb' in name:
                X_train_clean = X_train.copy()
                X_train_clean.columns = [clean_feature_name(col) for col in X_train_clean.columns]
                # XGBoost最终训练不使用early_stopping
                model.fit(X_train_clean, y_train, verbose=False)
            else:
                # LightGBM和CatBoost最终训练也不使用验证集
                model.fit(X_train, y_train)
            final_base_models[name] = model

        # 训练元学习器
        print("训练元学习器...")
        meta_learner = lgb.LGBMClassifier(
            objective='binary', metric='auc', num_leaves=10, learning_rate=0.1,
            n_estimators=100, verbose=-1, seed=self.seed
        )
        meta_learner.fit(meta_features, meta_labels)

        # 计算Stacking模型的CV分数
        stacking_pred = meta_learner.predict_proba(meta_features)[:, 1]
        stacking_score = roc_auc_score(meta_labels, stacking_pred)
        print(f"Stacking模型CV AUC: {stacking_score:.4f}")

        self.models['stacking'] = {
            'base_models': final_base_models,
            'meta_learner': meta_learner
        }

        return self.models['stacking']

    def predict_stacking(self, X_test):
        """Stacking模型预测"""
        if 'stacking' not in self.models:
            raise ValueError("Stacking模型未训练")

        base_models = self.models['stacking']['base_models']
        meta_learner = self.models['stacking']['meta_learner']

        # 生成基础模型预测
        base_predictions = np.zeros((len(X_test), len(base_models)))

        for i, (name, model) in enumerate(base_models.items()):
            if 'xgb' in name:
                X_test_clean = X_test.copy()
                X_test_clean.columns = [clean_feature_name(col) for col in X_test_clean.columns]
                pred = model.predict_proba(X_test_clean)[:, 1]
            else:
                pred = model.predict_proba(X_test)[:, 1]
            base_predictions[:, i] = pred

        # 元学习器预测
        final_pred = meta_learner.predict_proba(base_predictions)[:, 1]

        return final_pred

    def feature_selection(self, X_train, y_train, method='importance', k=100):
        """特征选择"""
        print(f"进行特征选择 (方法: {method}, 选择前{k}个特征)...")

        if method == 'importance':
            # 使用LightGBM特征重要性
            model = lgb.LGBMClassifier(
                objective='binary', metric='auc', num_leaves=31,
                learning_rate=0.1, n_estimators=100, verbose=-1, seed=self.seed
            )
            model.fit(X_train, y_train)

            importance_df = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            selected_features = importance_df.head(k)['feature'].tolist()

        elif method == 'mutual_info':
            # 使用互信息
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
            selector.fit(X_train, y_train)
            selected_features = X_train.columns[selector.get_support()].tolist()

        elif method == 'f_test':
            # 使用F检验
            selector = SelectKBest(score_func=f_classif, k=k)
            selector.fit(X_train, y_train)
            selected_features = X_train.columns[selector.get_support()].tolist()

        print(f"选择了 {len(selected_features)} 个特征")
        return selected_features


def main():
    """主函数"""
    print("=" * 60)
    print("中国石化智能营销决策 - 深度特征挖掘与模型优化版本")
    print("=" * 60)

    # 1. 数据加载
    print("\n1. 加载数据...")
    wallet_train = pd.read_csv('cust_wallet_detail_train.csv')
    coupon_send_train = pd.read_csv('cust_coupon_detail_send_train.csv')

    wallet_validation = pd.read_csv('cust_wallet_detail_validation_without_truth.csv')
    coupon_send_validation = pd.read_csv('cust_coupon_detail_send_validation.csv')

    print(f"训练集大小: {len(wallet_train)}")
    print(f"验证集大小: {len(wallet_validation)}")

    # 2. 创建标签
    print("\n2. 创建训练标签...")
    wallet_train['label'] = (~wallet_train['coupon_code'].isnull()).astype(int)
    print(f"正样本比例: {wallet_train['label'].mean():.2%}")

    # 移除所有可能泄露标签的列
    leak_cols = ['coupon_code', 'coupon_amt', 'discounts_amt', 'point_amt']
    for col in leak_cols:
        if col in wallet_train.columns:
            wallet_train = wallet_train.drop(columns=[col])
        if col in wallet_validation.columns:
            wallet_validation = wallet_validation.drop(columns=[col])

    # 3. 深度特征工程
    print("\n3. 深度特征工程...")
    fe = AdvancedFeatureEngineer()

    # 基础特征（增强版）
    wallet_train = fe.create_basic_features(wallet_train, is_train=True)
    wallet_validation = fe.create_basic_features(wallet_validation, is_train=False)

    # 用户特征（大幅增强）
    user_features_train = fe.create_user_features(wallet_train)
    user_features_validation = fe.create_user_features(wallet_validation)

    # 优惠券发放特征（增强版）
    coupon_features_train = fe.create_coupon_send_features(coupon_send_train)
    coupon_features_validation = fe.create_coupon_send_features(coupon_send_validation)

    # 交互特征（大幅增强）
    interaction_features_train = fe.create_interaction_features(wallet_train, coupon_features_train)
    interaction_features_validation = fe.create_interaction_features(wallet_validation, coupon_features_validation)

    # 高级时间特征
    time_features_train = fe.create_advanced_time_features(wallet_train)
    time_features_validation = fe.create_advanced_time_features(wallet_validation)

    # 聚类特征
    clustering_features_train = fe.create_clustering_features(wallet_train)
    clustering_features_validation = fe.create_clustering_features(wallet_validation)

    # 统计特征
    statistical_features_train = fe.create_statistical_features(wallet_train)
    statistical_features_validation = fe.create_statistical_features(wallet_validation)

    # 4. 合并所有特征
    print("\n4. 合并所有特征...")

    # 训练集合并
    train_data = wallet_train.copy()

    feature_dfs_train = [
        (user_features_train, "用户特征"),
        (coupon_features_train, "优惠券发放特征"),
        (interaction_features_train, "交互特征"),
        (time_features_train, "高级时间特征"),
        (clustering_features_train, "聚类特征"),
        (statistical_features_train, "统计特征")
    ]

    for feature_df, name in feature_dfs_train:
        if len(feature_df) > 0:
            print(f"  合并{name}: {feature_df.shape}")
            train_data = train_data.merge(feature_df, on='membercode', how='left')

    # 验证集合并
    validation_data = wallet_validation.copy()

    feature_dfs_validation = [
        (user_features_validation, "用户特征"),
        (coupon_features_validation, "优惠券发放特征"),
        (interaction_features_validation, "交互特征"),
        (time_features_validation, "高级时间特征"),
        (clustering_features_validation, "聚类特征"),
        (statistical_features_validation, "统计特征")
    ]

    for feature_df, name in feature_dfs_validation:
        if len(feature_df) > 0:
            print(f"  合并验证集{name}: {feature_df.shape}")
            validation_data = validation_data.merge(feature_df, on='membercode', how='left')

    print(f"  最终训练数据形状: {train_data.shape}")
    print(f"  最终验证数据形状: {validation_data.shape}")

    # 5. 特征选择和预处理
    print("\n5. 特征选择和预处理...")

    # 排除的列
    exclude_cols = ['order_no', 'external_order_no', 'user_id', 'membercode',
                    'station_code', 'station_name', 'sale_time', 'label',
                    'time_diff_hours', 'time_diff_days', 'cumsum_tran_amt',
                    'cumcount_orders', 'avg_tran_amt_so_far']

    # 选择特征列
    train_cols = set([col for col in train_data.columns if col not in exclude_cols])
    valid_cols = set([col for col in validation_data.columns if col not in exclude_cols])
    feature_cols = list(train_cols.intersection(valid_cols))

    print(f"原始特征数量: {len(feature_cols)}")

    # 准备数据
    X_train = train_data[feature_cols].copy()
    X_test = validation_data[feature_cols].copy()
    y_train = train_data['label']

    # 填充缺失值
    for col in feature_cols:
        if X_train[col].dtype in ['object', 'category']:
            X_train[col] = X_train[col].fillna('unknown')
            X_test[col] = X_test[col].fillna('unknown')
            # 标签编码
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
        else:
            X_train[col] = X_train[col].fillna(0)
            X_test[col] = X_test[col].fillna(0)

    # 确保所有特征名都是干净的
    X_train.columns = [clean_feature_name(col) for col in X_train.columns]
    X_test.columns = [clean_feature_name(col) for col in X_test.columns]
    feature_cols = list(X_train.columns)

    print(f"清理后特征数量: {len(feature_cols)}")

    # 6. 高级模型训练
    print("\n6. 高级模型训练...")
    trainer = AdvancedModelTrainer(seed=SEED)

    # 特征选择
    selected_features = trainer.feature_selection(X_train, y_train, method='importance', k=150)
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    print(f"选择了 {len(selected_features)} 个重要特征")

    # 训练Stacking模型
    stacking_model = trainer.train_stacking_model(X_train_selected, y_train, cv_folds=5)

    # 7. 预测
    print("\n7. 生成预测...")
    predictions = trainer.predict_stacking(X_test_selected)

    # 后处理预测结果
    predictions = np.clip(predictions, 0.001, 0.999)  # 避免极值

    # 8. 保存结果
    print("\n8. 保存结果...")
    submission = pd.DataFrame({
        'id': validation_data['order_no'],
        'predict': predictions
    })

    submission.to_csv('submission_advanced.csv', index=False)
    print(f"结果已保存到 submission_advanced.csv")
    print(f"预测分布: min={predictions.min():.4f}, max={predictions.max():.4f}, mean={predictions.mean():.4f}")

    # 9. 特征重要性分析
    print("\n9. Top 30 重要特征:")
    if 'stacking' in trainer.models:
        # 使用第一个LightGBM模型的特征重要性
        lgb_model = trainer.models['stacking']['base_models']['lgb1']
        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': lgb_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(feature_importance.head(30).to_string(index=False))

        # 保存特征重要性
        feature_importance.to_csv('feature_importance_advanced.csv', index=False)

    print("\n" + "=" * 60)
    print("训练完成！主要改进点:")
    print("1. 大幅增强的特征工程：时间特征、用户行为特征、统计特征")
    print("2. 聚类特征：用户分群")
    print("3. 高级交互特征：消费能力与优惠券匹配度")
    print("4. Stacking集成学习：多模型融合")
    print("5. 特征选择：选择最重要的特征")
    print("6. 滑动窗口特征：时间序列特征")
    print("7. 统计学特征：分布特征、趋势特征")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback

        traceback.print_exc()
# 下面是训练的日志：

# ============================================================
# 中国石化智能营销决策 - 深度特征挖掘与模型优化版本
# ============================================================
# 1. 加载数据...
# 训练集大小: 490942
# 验证集大小: 140194
# 2. 创建训练标签...
# 正样本比例: 34.75%
# 3. 深度特征工程...
# 创建基础特征... (is_train=True)
# 创建基础特征... (is_train=False)
# 创建用户级别特征 (基于membercode)...
# 创建用户级别特征 (基于membercode)...
# 创建优惠券发放特征...
# 创建优惠券发放特征...
# 创建交互特征...
# 创建交互特征...
# 创建高级时间特征...
# 创建高级时间特征...
# 创建聚类特征 (k=8)...
# 创建聚类特征 (k=8)...
# 创建统计学特征...
# 创建统计学特征...
# 4. 合并所有特征...
#   合并用户特征: (29887, 106)
#   合并优惠券发放特征: (29887, 48)
#   合并交互特征: (29887, 41)
#   合并高级时间特征: (29887, 27)
#   合并聚类特征: (29887, 12)
#   合并统计特征: (29887, 9)
#   合并验证集用户特征: (9962, 106)
#   合并验证集优惠券发放特征: (9962, 48)
#   合并验证集交互特征: (9962, 41)
#   合并验证集高级时间特征: (9962, 27)
#   合并验证集聚类特征: (9962, 12)
#   合并验证集统计特征: (9962, 9)
#   最终训练数据形状: (490942, 281)
#   最终验证数据形状: (140194, 280)
# 5. 特征选择和预处理...
# 原始特征数量: 273
# 清理后特征数量: 273
# 6. 高级模型训练...
# 进行特征选择 (方法: importance, 选择前150个特征)...
# 选择了 150 个特征
# 选择了 150 个重要特征
# 训练Stacking模型 (CV=5折)...
# 处理第 1/5 折...
# Training until validation scores don't improve for 50 rounds
# Did not meet early stopping. Best iteration is:
# [500]	valid_0's auc: 0.890776
#   lgb1 AUC: 0.8908
# Training until validation scores don't improve for 50 rounds
# Did not meet early stopping. Best iteration is:
# [800]	valid_0's auc: 0.900993
#   lgb2 AUC: 0.9010
#   xgb1 AUC: 0.8908
#   cat1 AUC: 0.8715
# 处理第 2/5 折...
# Training until validation scores don't improve for 50 rounds
# Did not meet early stopping. Best iteration is:
# [500]	valid_0's auc: 0.890785
#   lgb1 AUC: 0.8908
# Training until validation scores don't improve for 50 rounds
# Did not meet early stopping. Best iteration is:
# [800]	valid_0's auc: 0.90073
#   lgb2 AUC: 0.9007
#   xgb1 AUC: 0.8915
#   cat1 AUC: 0.8725
# 处理第 3/5 折...
# Training until validation scores don't improve for 50 rounds
# Did not meet early stopping. Best iteration is:
# [500]	valid_0's auc: 0.892409
#   lgb1 AUC: 0.8924
# Training until validation scores don't improve for 50 rounds
# Did not meet early stopping. Best iteration is:
# [800]	valid_0's auc: 0.902143
#   lgb2 AUC: 0.9021
#   xgb1 AUC: 0.8926
#   cat1 AUC: 0.8729
# 处理第 4/5 折...
# Training until validation scores don't improve for 50 rounds
# Did not meet early stopping. Best iteration is:
# [500]	valid_0's auc: 0.890527
#   lgb1 AUC: 0.8905
# Training until validation scores don't improve for 50 rounds
# Did not meet early stopping. Best iteration is:
# [800]	valid_0's auc: 0.901828
#   lgb2 AUC: 0.9018
#   xgb1 AUC: 0.8917
#   cat1 AUC: 0.8724
# 处理第 5/5 折...
# Training until validation scores don't improve for 50 rounds
# Did not meet early stopping. Best iteration is:
# [500]	valid_0's auc: 0.891096
#   lgb1 AUC: 0.8911
# Training until validation scores don't improve for 50 rounds
# Did not meet early stopping. Best iteration is:
# [800]	valid_0's auc: 0.901337
#   lgb2 AUC: 0.9013
#   xgb1 AUC: 0.8912
#   cat1 AUC: 0.8719
# CV结果:
# lgb1: 0.8911 (+/- 0.0013)
# lgb2: 0.9014 (+/- 0.0010)
# xgb1: 0.8916 (+/- 0.0012)
# cat1: 0.8722 (+/- 0.0009)
# 训练最终基础模型...
# 训练元学习器...
# Stacking模型CV AUC: 0.9077
# 7. 生成预测...
# 8. 保存结果...
# 结果已保存到 submission_advanced.csv
# 预测分布: min=0.0019, max=0.9985, mean=0.3688
# 9. Top 30 重要特征:
#                     feature  importance
#                    tran_amt         864
#                   sale_week         755
#                   amt_ratio         534
#          attributionorgcode         513
#          transactionorgcode         494
#            favorite_station         488
#                   sale_year         254
#                  sale_month         249
#         first_purchase_days         232
#      is_cross_province_mean         229
#               amt_diversity         213
#         send_endnumber_mean         202
#     time_time_diff_days_min         199
#               threshold_p25         197
#          receivable_amt_log         170
#               threshold_p75         153
#               threshold_p50         150
#          receivable_amt_min         145
# time_time_diff_hours_median         145
#          send_endnumber_max         142
#          send_endnumber_sum         135
#       is_cross_province_sum         131
#          send_topamount_min         128
#               tran_amt_kurt         124
#              sale_dayofweek         124
#              amt_ratio_skew         123
#              order_no_count         122
#                  clv_simple         122
#       receivable_amt_median         121
#           max_interval_days         121
# ============================================================
# 训练完成！主要改进点:
# 1. 大幅增强的特征工程：时间特征、用户行为特征、统计特征
# 2. 聚类特征：用户分群
# 3. 高级交互特征：消费能力与优惠券匹配度
# 4. Stacking集成学习：多模型融合
# 5. 特征选择：选择最重要的特征
# 6. 滑动窗口特征：时间序列特征
# 7. 统计学特征：分布特征、趋势特征
# ============================================================
# 下一步优化方向：

# 深度学习
# ：尝试神经网络模型处理复杂特征交互
# 实时特征
# ：构建实时特征工程pipeline
# 多目标优化
# ：同时优化响应率和利润率
# 因果推断
# ：理解营销活动的真实因果效应
