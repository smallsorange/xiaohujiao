# 中国石化人工智能大赛：基于测井曲线的岩性识别 - PSO优化版本
# 只使用 XGBoost 和 LightGBM，通过 PSO 进行超参数优化

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print(f"LightGBM version: {lgb.__version__}")
print(f"XGBoost version: {xgb.__version__}")

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)

set_seed(42)


# ==================== PSO 粒子群优化算法 ====================
class Particle:
    """粒子类"""
    def __init__(self, bounds):
        self.position = np.array([np.random.uniform(low, high) for low, high in bounds])
        self.velocity = np.random.uniform(-1, 1, len(bounds))
        self.best_position = self.position.copy()
        self.best_score = -np.inf
        self.score = -np.inf


class PSO:
    """粒子群优化算法"""
    def __init__(self, objective_func, bounds, n_particles=20, max_iter=30, w=0.7, c1=1.5, c2=1.5):
        """
        Args:
            objective_func: 目标函数（需要最大化）
            bounds: 参数边界 [(min1, max1), (min2, max2), ...]
            n_particles: 粒子数量
            max_iter: 最大迭代次数
            w: 惯性权重
            c1: 个体学习因子
            c2: 社会学习因子
        """
        self.objective_func = objective_func
        self.bounds = bounds
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # 初始化粒子群
        self.particles = [Particle(bounds) for _ in range(n_particles)]
        self.global_best_position = None
        self.global_best_score = -np.inf
        self.history = []
        
    def optimize(self):
        """执行PSO优化"""
        print(f"\n开始PSO优化: {self.n_particles}个粒子, {self.max_iter}次迭代")
        print("="*70)
        
        for iteration in range(self.max_iter):
            print(f"\n迭代 {iteration+1}/{self.max_iter}")
            
            # 评估每个粒子
            for i, particle in enumerate(self.particles):
                # 计算适应度
                score = self.objective_func(particle.position)
                particle.score = score
                
                # 更新个体最佳
                if score > particle.best_score:
                    particle.best_score = score
                    particle.best_position = particle.position.copy()
                
                # 更新全局最佳
                if score > self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = particle.position.copy()
                    print(f"  粒子 {i+1}: 新的最佳分数 = {score:.6f}")
            
            print(f"  当前全局最佳分数: {self.global_best_score:.6f}")
            self.history.append(self.global_best_score)
            
            # 更新粒子速度和位置
            for particle in self.particles:
                r1 = np.random.random(len(self.bounds))
                r2 = np.random.random(len(self.bounds))
                
                # 更新速度
                cognitive = self.c1 * r1 * (particle.best_position - particle.position)
                social = self.c2 * r2 * (self.global_best_position - particle.position)
                particle.velocity = self.w * particle.velocity + cognitive + social
                
                # 更新位置
                particle.position = particle.position + particle.velocity
                
                # 边界处理
                for j, (low, high) in enumerate(self.bounds):
                    if particle.position[j] < low:
                        particle.position[j] = low
                        particle.velocity[j] *= -0.5
                    elif particle.position[j] > high:
                        particle.position[j] = high
                        particle.velocity[j] *= -0.5
        
        print("\n" + "="*70)
        print(f"PSO优化完成!")
        print(f"最佳分数: {self.global_best_score:.6f}")
        print(f"最佳参数: {self.global_best_position}")
        
        return self.global_best_position, self.global_best_score


# ==================== 岩性识别系统 ====================
class LithologyIdentifier:
    """岩性识别系统（仅使用XGBoost和LightGBM）"""

    def __init__(self):
        self.models = {}
        self.feature_columns = []
        self.scaler = RobustScaler()
        self.best_params = {}

    def safe_divide(self, a, b, default=0):
        """安全除法"""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.divide(a, b)
            result = np.where(np.isfinite(result), result, default)
        return result

    def robust_clean_data(self, df):
        """数据清理"""
        df = df.copy()
        df = df.replace([np.inf, -np.inf], np.nan)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['WELL', 'DEPTH', 'label', 'ID', 'id']:
                Q1 = df[col].quantile(0.05)
                Q3 = df[col].quantile(0.95)
                IQR = Q3 - Q1
                lower_bound = Q1 - 2.5 * IQR
                upper_bound = Q3 + 2.5 * IQR
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                
                median_val = df[col].median()
                if not np.isfinite(median_val):
                    median_val = 0
                df[col] = df[col].fillna(median_val)
                df[col] = df[col].replace([np.inf, -np.inf], median_val)
        
        return df

    def create_advanced_features(self, df):
        """创建高级特征"""
        df = df.copy()
        basic_features = ['SP', 'GR', 'AC']

        # 1. 深度特征
        df['DEPTH_normalized'] = df.groupby('WELL')['DEPTH'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
        )
        df['DEPTH_diff'] = df.groupby('WELL')['DEPTH'].diff().fillna(0)
        df['DEPTH_diff2'] = df.groupby('WELL')['DEPTH_diff'].diff().fillna(0)
        df['DEPTH_zone'] = pd.cut(df['DEPTH'], bins=10, labels=False)
        df['DEPTH_zone_normalized'] = df['DEPTH_zone'] / 9.0

        # 2. 滑动窗口特征
        window_sizes = [3, 5, 7]
        for window in window_sizes:
            for feature in basic_features:
                if feature in df.columns:
                    df[f'{feature}_rolling_mean_{window}'] = df.groupby('WELL')[feature].transform(
                        lambda x: x.rolling(window=window, min_periods=1, center=True).mean()
                    )
                    df[f'{feature}_rolling_std_{window}'] = df.groupby('WELL')[feature].transform(
                        lambda x: x.rolling(window=window, min_periods=1, center=True).std().fillna(0)
                    )
                    df[f'{feature}_diff_{window}'] = df.groupby('WELL')[feature].transform(
                        lambda x: x.diff(window).fillna(0)
                    )

        # 3. 地质物理特征
        if 'GR' in df.columns:
            df['GR_min'] = df.groupby('WELL')['GR'].transform('min')
            df['GR_max'] = df.groupby('WELL')['GR'].transform('max')
            df['Vsh_linear'] = self.safe_divide(df['GR'] - df['GR_min'], df['GR_max'] - df['GR_min'] + 1e-8)
            df['Vsh_linear'] = np.clip(df['Vsh_linear'], 0, 1)
            
            IGR = df['Vsh_linear']
            df['Vsh_larionov'] = 0.083 * (2**(3.7 * IGR) - 1)
            df['Vsh_larionov'] = np.clip(df['Vsh_larionov'], 0, 1)
            
            df['Vsh_steiber'] = self.safe_divide(IGR, 3 - 2 * IGR)
            df['Vsh_steiber'] = np.clip(df['Vsh_steiber'], 0, 1)
            
            df['GR_normalized'] = self.safe_divide(
                df['GR'] - df.groupby('WELL')['GR'].transform('median'),
                df.groupby('WELL')['GR'].transform('std') + 1e-8
            )

        if 'AC' in df.columns:
            df['PHI_AC'] = self.safe_divide(df['AC'] - 180, 300 - 180)
            df['PHI_AC'] = np.clip(df['PHI_AC'], 0, 1)
            
            AC_matrix = 182
            AC_fluid = 617
            df['PHI_wyllie'] = self.safe_divide(df['AC'] - AC_matrix, AC_fluid - AC_matrix)
            df['PHI_wyllie'] = np.clip(df['PHI_wyllie'], 0, 0.4)
            
            df['AC_normalized'] = self.safe_divide(
                df['AC'] - df.groupby('WELL')['AC'].transform('median'),
                df.groupby('WELL')['AC'].transform('std') + 1e-8
            )

        if 'Vsh_linear' in df.columns and 'PHI_AC' in df.columns:
            df['PHI_density_proxy'] = 1 - df['Vsh_linear']
            df['PHI_effective'] = df['PHI_AC'] * (1 - df['Vsh_linear'])
            df['permeability_proxy'] = self.safe_divide(
                df['PHI_effective'] ** 3,
                (1 - df['PHI_effective']) ** 2 + 1e-8
            )
            
        if 'SP' in df.columns:
            df['SP_normalized'] = self.safe_divide(
                df['SP'] - df.groupby('WELL')['SP'].transform('median'),
                df.groupby('WELL')['SP'].transform('std') + 1e-8
            )
            df['SP_baseline_deviation'] = df['SP'] - df.groupby('WELL')['SP'].transform('mean')

        # 4. 特征交互
        if all(col in df.columns for col in ['GR', 'AC']):
            df['GR_AC_ratio'] = self.safe_divide(df['GR'], df['AC'] + 1e-8)
            df['GR_AC_product'] = df['GR'] * df['AC']
            df['GR_AC_diff'] = df['GR'] - df['AC']

        if all(col in df.columns for col in ['SP', 'GR']):
            df['SP_GR_ratio'] = self.safe_divide(df['SP'], df['GR'] + 1e-8)
            df['SP_GR_product'] = df['SP'] * df['GR']
            df['SP_GR_diff'] = df['SP'] - df['GR']

        if all(col in df.columns for col in ['SP', 'AC']):
            df['SP_AC_ratio'] = self.safe_divide(df['SP'], df['AC'] + 1e-8)
            df['SP_AC_product'] = df['SP'] * df['AC']
            df['SP_AC_diff'] = df['SP'] - df['AC']
            
        if all(col in df.columns for col in ['SP', 'GR', 'AC']):
            df['sand_shale_index'] = self.safe_divide(df['SP'] * df['AC'], df['GR'] + 1e-8)
            df['lithology_index'] = self.safe_divide(df['GR'] * df['AC'], np.abs(df['SP']) + 1e-8)

        # 5. 数学变换
        for feature in basic_features:
            if feature in df.columns:
                df[f'{feature}_log'] = np.log1p(np.maximum(0, df[feature] - df[feature].min() + 1e-8))
                df[f'{feature}_sqrt'] = np.sqrt(np.maximum(0, df[feature] - df[feature].min() + 1e-8))

        # 6. 排序特征
        for feature in basic_features:
            if feature in df.columns:
                df[f'{feature}_rank'] = df.groupby('WELL')[feature].rank(pct=True)

        # 7. 井内标准化
        for feature in basic_features:
            if feature in df.columns:
                df[f'{feature}_well_normalized'] = df.groupby('WELL')[feature].transform(
                    lambda x: self.safe_divide(x - x.mean(), x.std() + 1e-8)
                )
                
        # 8. 梯度特征
        for feature in basic_features:
            if feature in df.columns:
                df[f'{feature}_gradient'] = df.groupby('WELL')[feature].diff().fillna(0)
                df[f'{feature}_gradient2'] = df.groupby('WELL')[f'{feature}_gradient'].diff().fillna(0)
                df[f'{feature}_gradient_pct'] = df.groupby('WELL')[feature].pct_change().fillna(0)
                df[f'{feature}_gradient_sign'] = np.sign(df[f'{feature}_gradient'])
        
        # 9. 波动性特征
        for feature in basic_features:
            if feature in df.columns:
                rolling_mean = df.groupby('WELL')[feature].transform(
                    lambda x: x.rolling(7, min_periods=1, center=True).mean()
                )
                rolling_std = df.groupby('WELL')[feature].transform(
                    lambda x: x.rolling(7, min_periods=1, center=True).std().fillna(0)
                )
                df[f'{feature}_cv'] = self.safe_divide(rolling_std, rolling_mean + 1e-8)
                
                df[f'{feature}_is_local_max'] = (
                    df.groupby('WELL')[feature].transform(
                        lambda x: x.rolling(5, min_periods=1, center=True).apply(
                            lambda y: 1.0 if len(y) > 0 and y.iloc[len(y)//2] == y.max() else 0.0
                        )
                    )
                )
                df[f'{feature}_is_local_min'] = (
                    df.groupby('WELL')[feature].transform(
                        lambda x: x.rolling(5, min_periods=1, center=True).apply(
                            lambda y: 1.0 if len(y) > 0 and y.iloc[len(y)//2] == y.min() else 0.0
                        )
                    )
                )
        
        # 10. 峰峰值特征
        for feature in basic_features:
            if feature in df.columns:
                df[f'{feature}_peak_to_peak'] = df.groupby('WELL')[feature].transform(
                    lambda x: x.rolling(11, min_periods=1, center=True).apply(
                        lambda y: y.max() - y.min() if len(y) > 0 else 0
                    )
                )

        # 删除临时列
        df = df.drop(['GR_min', 'GR_max'], axis=1, errors='ignore')
        df = self.robust_clean_data(df)

        return df

    def pso_optimize_lgb(self, X_train, y_train, groups, n_particles=15, max_iter=20):
        """使用PSO优化LightGBM参数"""
        print("\n" + "="*70)
        print("PSO优化 LightGBM 超参数")
        print("="*70)
        
        # 定义参数边界 [learning_rate, num_leaves, max_depth, min_child_samples, subsample, colsample_bytree, reg_alpha, reg_lambda]
        bounds = [
            (0.005, 0.1),    # learning_rate
            (20, 80),        # num_leaves
            (4, 12),         # max_depth
            (10, 100),       # min_child_samples
            (0.5, 1.0),      # subsample
            (0.5, 1.0),      # colsample_bytree
            (0.0, 1.0),      # reg_alpha
            (0.0, 1.0),      # reg_lambda
        ]
        
        # 准备交叉验证
        n_groups = len(np.unique(groups))
        n_splits = min(4, n_groups)
        gkf = GroupKFold(n_splits=n_splits) if n_splits > 1 else StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        
        def objective(params):
            """目标函数：返回交叉验证F1分数"""
            learning_rate, num_leaves, max_depth, min_child_samples, subsample, colsample_bytree, reg_alpha, reg_lambda = params
            
            lgb_params = {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'learning_rate': float(learning_rate),
                'num_leaves': int(num_leaves),
                'max_depth': int(max_depth),
                'min_child_samples': int(min_child_samples),
                'subsample': float(subsample),
                'colsample_bytree': float(colsample_bytree),
                'reg_alpha': float(reg_alpha),
                'reg_lambda': float(reg_lambda),
                'random_state': 42,
                'n_estimators': 500,
                'verbose': -1
            }
            
            scores = []
            splits = list(gkf.split(X_train, y_train, groups if n_splits > 1 else None))
            
            for train_idx, val_idx in splits[:2]:  # 只用前2折加速
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                try:
                    model = lgb.LGBMClassifier(**lgb_params)
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
                    pred = model.predict(X_val)
                    score = f1_score(y_val, pred, average='macro')
                    scores.append(score)
                except:
                    scores.append(0.0)
            
            return np.mean(scores)
        
        # 执行PSO优化
        pso = PSO(objective, bounds, n_particles=n_particles, max_iter=max_iter)
        best_params, best_score = pso.optimize()
        
        # 转换为字典
        self.best_params['lgb'] = {
            'learning_rate': float(best_params[0]),
            'num_leaves': int(best_params[1]),
            'max_depth': int(best_params[2]),
            'min_child_samples': int(best_params[3]),
            'subsample': float(best_params[4]),
            'colsample_bytree': float(best_params[5]),
            'reg_alpha': float(best_params[6]),
            'reg_lambda': float(best_params[7]),
        }
        
        print(f"\nLightGBM 最佳参数: {self.best_params['lgb']}")
        return best_score

    def pso_optimize_xgb(self, X_train, y_train, groups, n_particles=15, max_iter=20):
        """使用PSO优化XGBoost参数"""
        print("\n" + "="*70)
        print("PSO优化 XGBoost 超参数")
        print("="*70)
        
        # 定义参数边界 [learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, gamma, reg_alpha, reg_lambda]
        bounds = [
            (0.005, 0.1),    # learning_rate
            (3, 12),         # max_depth
            (1, 20),         # min_child_weight
            (0.5, 1.0),      # subsample
            (0.5, 1.0),      # colsample_bytree
            (0.0, 1.0),      # gamma
            (0.0, 1.0),      # reg_alpha
            (0.0, 1.0),      # reg_lambda
        ]
        
        # 准备交叉验证
        n_groups = len(np.unique(groups))
        n_splits = min(4, n_groups)
        gkf = GroupKFold(n_splits=n_splits) if n_splits > 1 else StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        
        def objective(params):
            """目标函数：返回交叉验证F1分数"""
            learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, gamma, reg_alpha, reg_lambda = params
            
            xgb_params = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'learning_rate': float(learning_rate),
                'max_depth': int(max_depth),
                'min_child_weight': float(min_child_weight),
                'subsample': float(subsample),
                'colsample_bytree': float(colsample_bytree),
                'gamma': float(gamma),
                'reg_alpha': float(reg_alpha),
                'reg_lambda': float(reg_lambda),
                'random_state': 42,
                'n_estimators': 500,
                'tree_method': 'hist',
                'eval_metric': 'mlogloss',
                'verbosity': 0,
            }
            
            scores = []
            splits = list(gkf.split(X_train, y_train, groups if n_splits > 1 else None))
            
            for train_idx, val_idx in splits[:2]:  # 只用前2折加速
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                try:
                    model = xgb.XGBClassifier(**xgb_params)
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                    pred = model.predict(X_val)
                    score = f1_score(y_val, pred, average='macro')
                    scores.append(score)
                except:
                    scores.append(0.0)
            
            return np.mean(scores)
        
        # 执行PSO优化
        pso = PSO(objective, bounds, n_particles=n_particles, max_iter=max_iter)
        best_params, best_score = pso.optimize()
        
        # 转换为字典
        self.best_params['xgb'] = {
            'learning_rate': float(best_params[0]),
            'max_depth': int(best_params[1]),
            'min_child_weight': float(best_params[2]),
            'subsample': float(best_params[3]),
            'colsample_bytree': float(best_params[4]),
            'gamma': float(best_params[5]),
            'reg_alpha': float(best_params[6]),
            'reg_lambda': float(best_params[7]),
        }
        
        print(f"\nXGBoost 最佳参数: {self.best_params['xgb']}")
        return best_score

    def train_optimized_models(self, X_train, y_train, groups):
        """使用PSO优化后的参数训练模型"""
        print("\n" + "="*70)
        print("使用优化参数训练最终模型")
        print("="*70)
        
        X_train_clean = self.robust_clean_data(X_train)
        
        models = {}
        n_groups = len(np.unique(groups))
        n_splits = min(5, n_groups)
        
        if n_splits > 1:
            gkf = GroupKFold(n_splits=n_splits)
            splits = list(gkf.split(X_train_clean, y_train, groups=groups))
        else:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            splits = list(skf.split(X_train_clean, y_train))

        # 训练 LightGBM
        print("\n训练 LightGBM 模型...")
        lgb_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'random_state': 42,
            'n_estimators': 2000,
            'verbose': -1,
            **self.best_params['lgb']
        }
        
        lgb_models = []
        lgb_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            X_tr, X_val = X_train_clean.iloc[train_idx], X_train_clean.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = lgb.LGBMClassifier(**lgb_params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(150), lgb.log_evaluation(100)])
            
            pred = model.predict(X_val)
            score = f1_score(y_val, pred, average='macro')
            lgb_scores.append(score)
            lgb_models.append(model)
            print(f"  Fold {fold+1} F1: {score:.4f}")
        
        models['lgb'] = {'models': lgb_models, 'cv_score': np.mean(lgb_scores)}
        print(f"LightGBM 平均CV分数: {np.mean(lgb_scores):.4f}")

        # 训练 XGBoost
        print("\n训练 XGBoost 模型...")
        xgb_params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'random_state': 42,
            'n_estimators': 2000,
            'tree_method': 'hist',
            'eval_metric': 'mlogloss',
            'verbosity': 0,
            'early_stopping_rounds': 150,
            **self.best_params['xgb']
        }
        
        xgb_models = []
        xgb_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            X_tr, X_val = X_train_clean.iloc[train_idx], X_train_clean.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = xgb.XGBClassifier(**xgb_params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            
            pred = model.predict(X_val)
            score = f1_score(y_val, pred, average='macro')
            xgb_scores.append(score)
            xgb_models.append(model)
            print(f"  Fold {fold+1} F1: {score:.4f}")
        
        models['xgb'] = {'models': xgb_models, 'cv_score': np.mean(xgb_scores)}
        print(f"XGBoost 平均CV分数: {np.mean(xgb_scores):.4f}")

        return models

    def ensemble_predictions(self, models, X_test):
        """集成预测"""
        print("\n进行集成预测...")
        
        X_test_clean = self.robust_clean_data(X_test)
        
        all_predictions = []
        weights = []
        
        for model_name, model_info in models.items():
            fold_preds = []
            for model in model_info['models']:
                pred = model.predict_proba(X_test_clean)
                fold_preds.append(pred)
            
            avg_pred = np.mean(fold_preds, axis=0)
            all_predictions.append(avg_pred)
            weights.append(model_info['cv_score'])
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        print(f"模型权重: LightGBM={weights[0]:.4f}, XGBoost={weights[1]:.4f}")
        
        ensemble_pred = np.zeros_like(all_predictions[0])
        for pred, weight in zip(all_predictions, weights):
            ensemble_pred += pred * weight
        
        final_predictions = np.argmax(ensemble_pred, axis=1)
        
        return final_predictions

    def train_and_predict(self, train_path, test_path, optimize=True, n_particles=15, max_iter=20):
        """主训练预测流程"""
        # 加载数据
        print("加载数据...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        print(f"训练数据形状: {train_df.shape}")
        print(f"测试数据形状: {test_df.shape}")
        if 'label' in train_df.columns:
            print(f"训练数据岩性分布:\n{train_df['label'].value_counts().sort_index()}")
        
        # 特征工程
        print("\n进行特征工程...")
        train_fe = self.create_advanced_features(train_df)
        test_fe = self.create_advanced_features(test_df)
        
        # 选择特征列
        non_feature_cols = ['WELL', 'DEPTH', 'label', 'ID', 'id']
        self.feature_columns = [col for col in train_fe.columns 
                               if col not in non_feature_cols 
                               and col in test_fe.columns
                               and train_fe[col].dtype in ['float64', 'int64']]
        
        print(f"特征数量: {len(self.feature_columns)}")
        
        # 准备数据
        X_train = train_fe[self.feature_columns]
        y_train = train_fe['label']
        groups = train_fe['WELL']
        X_test = test_fe[self.feature_columns]
        
        print(f"训练数据特征形状: {X_train.shape}")
        
        X_train = self.robust_clean_data(X_train)
        X_test = self.robust_clean_data(X_test)
        
        # PSO优化超参数
        if optimize:
            print("\n" + "="*70)
            print("开始 PSO 超参数优化")
            print("="*70)
            self.pso_optimize_lgb(X_train, y_train, groups, n_particles, max_iter)
            self.pso_optimize_xgb(X_train, y_train, groups, n_particles, max_iter)
        else:
            # 使用默认参数
            self.best_params = {
                'lgb': {
                    'learning_rate': 0.02,
                    'num_leaves': 31,
                    'max_depth': 8,
                    'min_child_samples': 20,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.1,
                },
                'xgb': {
                    'learning_rate': 0.02,
                    'max_depth': 8,
                    'min_child_weight': 3,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'gamma': 0.1,
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.1,
                }
            }
        
        # 训练模型
        models = self.train_optimized_models(X_train, y_train, groups)
        
        # 预测
        test_predictions = self.ensemble_predictions(models, X_test)
        
        # 创建提交文件
        id_column = 'ID' if 'ID' in test_df.columns else 'id'
        submission = pd.DataFrame({
            'id': test_df[id_column].values,
            'predict': test_predictions
        })
        
        return submission


# 主程序
if __name__ == "__main__":
    identifier = LithologyIdentifier()
    
    try:
        # optimize=True: 使用PSO优化, False: 使用默认参数
        # n_particles: PSO粒子数 (建议10-20)
        # max_iter: PSO迭代次数 (建议15-30)
        submission = identifier.train_and_predict(
            "train.csv",
            "validation_without_label.csv",
            optimize=True,      # 是否启用PSO优化
            n_particles=15,     # PSO粒子数
            max_iter=20         # PSO最大迭代次数
        )
        
        if submission is not None:
            submission.to_csv("submission_pso.csv", index=False)
            print("\n" + "="*70)
            print("预测完成！结果已保存到 submission_pso.csv")
            print("="*70)
            print(f"提交文件形状: {submission.shape}")
            print(f"预测值分布:\n{submission['predict'].value_counts().sort_index()}")
        else:
            print("训练失败，无法生成预测结果")
    
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()
