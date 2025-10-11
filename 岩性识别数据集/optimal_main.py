# 中国石化人工智能大赛：基于测井曲线的岩性识别与分类算法0.75566分的Baseline方案完整代码分享
# 比赛的地址：
# https://aicup.sinopec.com/competition/SINOPEC-01/

# 背景介绍
# 在油气勘探与开发领域，地层岩性（即岩石类型）的精确识别是资源评价、储层预测和钻井工程设计的基础。测井技术通过在钻孔中下放精密仪器，连续测量地层沿井深的各种地球物理参数（如电学、声学、放射性等），形成了丰富的测井曲线数据。这些曲线蕴含着关于地下岩石类型、物性（孔隙度、渗透率）及流体性质（油、气、水）的关键信息。
 
# 传统上，岩性识别主要依赖于地质工程师的人工解释。该方法不仅耗时耗力，而且解释结果高度依赖于个人经验，难以实现标准化和规模化。随着人工智能技术的发展，利用机器学习算法分析多维度的测井曲线，能够自动、高效、精准地识别岩性，已经成为地球物理勘探领域的研究热点。
 
# 本次竞赛旨在鼓励参赛者探索和应用前沿的人工智能技术，通过分析真实的测井数据，构建一个高精度的岩性自动识别模型。这不仅能极大提升地质研究的效率和准确性，还能为油气资源的智能化勘探与开发提供坚实的技术支撑，具有重要的学术价值和广阔的工业应用前景。

# 竞赛任务

# 选手需要利用提供的测井曲线数据集，构建一个或多个高精度的机器学习或深度学习模型，对给定深度点的地层岩性进行自动识别和分类。具体任务是将每个深度点的岩性准确地分类为砂

# 岩、粉砂岩、泥岩三类之一。

# 目前暂时第16名：

# 图片
# 下面是0.75566分的完整baseline代码：
# 在之前0.73096分的方案的基础上，将lgb和xgb的max_depth增加到16，将catboost的max_depth增加到8，然后借鉴了“算法指难针”的大佬“匀速小子”的一个后处理操作：对预测值为0和2的样本中，选择label=1的概率值大于某个阈值（本文用的0.45的阈值）的样本将他们改为1。经过两个综合操作然后线上分数就从0.73096分涨到0.75566分。

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from scipy import stats
from scipy.signal import savgol_filter
import warnings

warnings.filterwarnings('ignore')


# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device(
    'cuda' if torch.cuda.is_available() 
    else 'mps' if torch.backends.mps.is_available() 
    else 'cpu'
)
print(f"Using device: {device}")


class WellLogDataset(Dataset):
    """测井数据集"""

    def __init__(self, features, labels=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels) if labels is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]


class LSTMModel(nn.Module):
    """LSTM模型用于序列特征学习"""

    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=3, dropout=0.2):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = x.unsqueeze(1)

        lstm_out, _ = self.lstm(x)

        attention_weights = self.attention(lstm_out)
        attended_features = torch.sum(lstm_out * attention_weights, dim=1)

        output = self.classifier(attended_features)
        return output


class TransformerBlock(nn.Module):
    """增强的Transformer块"""

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        return x


class EnhancedTabularTransformer(nn.Module):
    """增强的表格数据Transformer"""

    def __init__(self, input_dim, num_classes=3, embed_dim=256, num_heads=16,
                 num_layers=6, ff_dim=512, dropout=0.2):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.positional_encoding = nn.Parameter(torch.randn(1, 100, embed_dim))

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.LayerNorm(embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, num_classes)
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = x.unsqueeze(1)

        x = x + self.positional_encoding[:, :x.size(1), :]

        for transformer in self.transformer_blocks:
            x = transformer(x)

        x = x.mean(dim=1)

        output = self.classifier(x)
        return output


class EnhancedCNN1DModel(nn.Module):
    """增强的1D CNN模型"""

    def __init__(self, input_dim, num_classes=3, dropout=0.3):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            nn.Conv1d(128, 256, kernel_size=7, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_blocks(x)
        x = x.squeeze(-1)
        output = self.classifier(x)
        return output


class UltraFastLithologyIdentifier:
    """超快速岩性识别系统 - 使用高效融合策略"""

    def __init__(self):
        self.models = {}
        self.feature_columns = []
        self.scalers = {
            'robust': RobustScaler(),
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'power': PowerTransformer()
        }
        self.best_epochs = {}

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
                Q1 = df[col].quantile(0.01)
                Q3 = df[col].quantile(0.99)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR

                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                median_val = df[col].median()
                if not np.isfinite(median_val):
                    median_val = 0
                df[col] = df[col].fillna(median_val)
                df[col] = df[col].replace([np.inf, -np.inf], median_val)

        return df

    def create_ultra_features(self, df):
        """创建超级特征（简化版本，保留最有效的特征）"""
        df = df.copy()
        basic_features = ['SP', 'GR', 'AC']

        # 确保基础特征是数值型
        for feature in basic_features:
            if feature in df.columns:
                df[feature] = pd.to_numeric(df[feature], errors='coerce')

        # 初始数据清理
        df = self.robust_clean_data(df)

        print("创建核心特征...")
        # 1. 深度特征
        df['DEPTH_normalized'] = df.groupby('WELL')['DEPTH'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
        )
        df['DEPTH_squared'] = df['DEPTH_normalized'] ** 2
        df['DEPTH_sqrt'] = np.sqrt(df['DEPTH_normalized'])

        print("创建滑窗特征...")
        # 2. 核心滑窗特征（增加窗口数量）
        window_sizes = list(range(1, 52))  # 增加窗口数量

        for window in window_sizes:
            for feature in basic_features:
                if feature in df.columns:
                    # 主要统计量
                    df[f'{feature}_roll_mean_{window}'] = df.groupby('WELL')[feature].transform(
                        lambda x: x.rolling(window=window, min_periods=1, center=True).mean()
                    )
                    df[f'{feature}_roll_std_{window}'] = df.groupby('WELL')[feature].transform(
                        lambda x: x.rolling(window=window, min_periods=1, center=True).std().fillna(0)
                    )
                    df[f'{feature}_roll_median_{window}'] = df.groupby('WELL')[feature].transform(
                        lambda x: x.rolling(window=window, min_periods=1, center=True).median()
                    )

                    # 相对值特征
                    df[f'{feature}_vs_roll_mean_{window}'] = self.safe_divide(
                        df[feature], df[f'{feature}_roll_mean_{window}'] + 1e-8
                    )

                    # Z-score特征
                    df[f'{feature}_zscore_{window}'] = self.safe_divide(
                        df[feature] - df[f'{feature}_roll_mean_{window}'],
                        df[f'{feature}_roll_std_{window}'] + 1e-8
                    )

        print("创建差分特征...")
        # 3. 差分特征
        for feature in basic_features:
            if feature in df.columns:
                for lag in list(range(1, 16)):
                    df[f'{feature}_diff_{lag}'] = df.groupby('WELL')[feature].diff(lag).fillna(0)
                    df[f'{feature}_pct_{lag}'] = df.groupby('WELL')[feature].pct_change(lag).fillna(0)

        print("创建地质物理特征...")
        # 4. 地质物理特征
        if 'GR' in df.columns:
            df['GR_min'] = df.groupby('WELL')['GR'].transform('min')
            df['GR_max'] = df.groupby('WELL')['GR'].transform('max')

            df['Vsh_linear'] = self.safe_divide(df['GR'] - df['GR_min'], df['GR_max'] - df['GR_min'])
            df['Vsh_linear'] = np.clip(df['Vsh_linear'], 0, 1)

            df['Vsh_larionov_old'] = 0.33 * (2 ** (2 * df['Vsh_linear']) - 1)
            df['Vsh_larionov_tertiary'] = 0.083 * (2 ** (3.7 * df['Vsh_linear']) - 1)

        if 'AC' in df.columns:
            df['PHI_wyllie'] = self.safe_divide(df['AC'] - 180, 300 - 180)
            df['PHI_raymer'] = 0.67 * self.safe_divide(df['AC'] - 180, df['AC'])
            df['PHI_wyllie'] = np.clip(df['PHI_wyllie'], 0, 1)
            df['PHI_raymer'] = np.clip(df['PHI_raymer'], 0, 1)

        print("创建交叉特征...")
        # 5. 特征交叉
        feature_pairs = [('GR', 'AC'), ('GR', 'SP'), ('AC', 'SP')]

        for f1, f2 in feature_pairs:
            if f1 in df.columns and f2 in df.columns:
                df[f'{f1}_{f2}_product'] = df[f1] * df[f2]
                df[f'{f1}_{f2}_ratio'] = self.safe_divide(df[f1], df[f2] + 1e-8)
                df[f'{f1}_{f2}_mean'] = (df[f1] + df[f2]) / 2

        # 6. 多项式特征
        for feature in basic_features:
            if feature in df.columns:
                df[f'{feature}_squared'] = df[feature] ** 2
                df[f'{feature}_sqrt'] = np.sqrt(np.abs(df[feature]))
                df[f'{feature}_log'] = np.log1p(np.maximum(0, df[feature] - df[feature].min() + 1e-8))

        # 7. 排序特征
        for feature in basic_features:
            if feature in df.columns:
                df[f'{feature}_rank'] = df.groupby('WELL')[feature].rank(pct=True)

        # 清理临时列
        df = df.drop(['GR_min', 'GR_max'], axis=1, errors='ignore')

        print("特征工程完成，进行最终清理...")
        # 最终数据清理
        df = self.robust_clean_data(df)

        print(
            f"特征工程完成，总特征数: {len([col for col in df.columns if col not in ['WELL', 'DEPTH', 'label', 'ID', 'id']])}")
        return df

    def train_tree_models(self, X_train, y_train, X_val, y_val):
        """训练树模型（XGBoost, LightGBM, CatBoost）"""
        print("\n训练树模型...")

        # 合并训练和验证集
        X_full = pd.concat([X_train, X_val], axis=0)
        y_full = pd.concat([y_train, y_val], axis=0)

        # 数据清理
        X_full = self.robust_clean_data(X_full)

        models = {}

        # LightGBM
        print("训练LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            objective='multiclass', num_class=3, learning_rate=0.02,
            num_leaves=31, max_depth=16, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
            random_state=42, n_estimators=500, verbose=-1
        )
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        val_pred = lgb_model.predict(X_val)
        val_f1 = f1_score(y_val, val_pred, average='macro')
        models['lgb'] = {'model': lgb_model, 'score': val_f1}
        print(f"LightGBM验证F1: {val_f1:.4f}")

        # XGBoost
        print("训练XGBoost...")
        xgb_model = xgb.XGBClassifier(
            objective='multi:softprob', num_class=3, learning_rate=0.02,
            max_depth=16, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, random_state=42,
            n_estimators=500, tree_method='hist', verbosity=0,
            early_stopping_rounds=100
        )
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        val_pred = xgb_model.predict(X_val)
        val_f1 = f1_score(y_val, val_pred, average='macro')
        models['xgb'] = {'model': xgb_model, 'score': val_f1}
        print(f"XGBoost验证F1: {val_f1:.4f}")

        # CatBoost
        print("训练CatBoost...")
        cat_model = CatBoostClassifier(
            loss_function='MultiClass', learning_rate=0.03, depth=8,
            l2_leaf_reg=5, random_seed=42, iterations=500,
            early_stopping_rounds=100, verbose=False
        )
        cat_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        val_pred = cat_model.predict(X_val)
        val_f1 = f1_score(y_val, val_pred, average='macro')
        models['cat'] = {'model': cat_model, 'score': val_f1}
        print(f"CatBoost验证F1: {val_f1:.4f}")

        # Extra Trees（额外的集成模型）
        print("训练Extra Trees...")
        et_model = ExtraTreesClassifier(
            n_estimators=300, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )
        et_model.fit(X_full, y_full)
        val_pred = et_model.predict(X_val)
        val_f1 = f1_score(y_val, val_pred, average='macro')
        models['et'] = {'model': et_model, 'score': val_f1}
        print(f"Extra Trees验证F1: {val_f1:.4f}")

        # Random Forest
        print("训练Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )
        rf_model.fit(X_full, y_full)
        val_pred = rf_model.predict(X_val)
        val_f1 = f1_score(y_val, val_pred, average='macro')
        models['rf'] = {'model': rf_model, 'score': val_f1}
        print(f"Random Forest验证F1: {val_f1:.4f}")

        return models

    def train_neural_models(self, X_train, y_train, X_val, y_val):
        """训练神经网络模型"""
        print("\n训练神经网络模型...")

        models = {}
        input_dim = X_train.shape[1]

        # 数据标准化
        scalers_fitted = {}
        for scaler_name, scaler in self.scalers.items():
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)

            scalers_fitted[scaler_name] = {
                'scaler': scaler,
                'X_train': X_train_scaled,
                'X_val': X_val_scaled
            }

        # 神经网络配置
        nn_configs = [
            ('transformer', EnhancedTabularTransformer(input_dim, num_classes=3,
                                                       embed_dim=128, num_heads=8,
                                                       num_layers=3, ff_dim=256, dropout=0.2), 'robust'),
            ('cnn', EnhancedCNN1DModel(input_dim, num_classes=3, dropout=0.3), 'standard'),
            ('lstm', LSTMModel(input_dim, hidden_dim=64, num_layers=2,
                               num_classes=3, dropout=0.2), 'minmax')
        ]

        for model_name, model_class, scaler_name in nn_configs:
            print(f"训练 {model_name}...")

            X_train_scaled = scalers_fitted[scaler_name]['X_train']
            X_val_scaled = scalers_fitted[scaler_name]['X_val']

            # 合并数据用于最终训练
            X_full = np.vstack([X_train_scaled, X_val_scaled])
            y_full = np.hstack([y_train.values, y_val.values])

            # 创建数据加载器
            train_dataset = WellLogDataset(X_train_scaled, y_train.values)
            val_dataset = WellLogDataset(X_val_scaled, y_val.values)
            full_dataset = WellLogDataset(X_full, y_full)

            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
            full_loader = DataLoader(full_dataset, batch_size=256, shuffle=True)

            model = model_class.to(device)

            # 训练参数
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

            # 快速训练找最佳epoch
            best_epoch, best_f1 = self.find_best_epoch_fast(
                model, train_loader, val_loader, criterion, optimizer, max_epochs=30
            )

            # 在全量数据上训练
            model = model_class.to(device)
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            scheduler = OneCycleLR(optimizer, max_lr=0.003, epochs=best_epoch + 1,
                                   steps_per_epoch=len(full_loader))

            for epoch in range(best_epoch + 1):
                model.train()
                for batch_features, batch_labels in full_loader:
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()

            models[model_name] = {
                'model': model,
                'score': best_f1,
                'scaler': scalers_fitted[scaler_name]['scaler']
            }
            print(f"{model_name}验证F1: {best_f1:.4f}")

        return models

    def find_best_epoch_fast(self, model, train_loader, val_loader, criterion, optimizer, max_epochs=30):
        """快速寻找最佳epoch"""
        best_val_f1 = 0
        best_epoch = 0
        patience = 50
        patience_counter = 0

        for epoch in range(max_epochs):
            # 训练
            model.train()
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)

                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # 验证
            model.eval()
            val_preds = []
            val_labels = []

            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(device)
                    outputs = model(batch_features)
                    val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                    val_labels.extend(batch_labels.cpu().numpy())

            val_f1 = f1_score(val_labels, val_preds, average='macro')

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        return best_epoch, best_val_f1

    def weighted_average_ensemble(self, models, X_test):
        """加权平均集成预测 - 修改：返回概率和预测"""
        print("\n进行加权平均集成预测...")

        # 数据清理
        X_test_clean = self.robust_clean_data(X_test.copy())

        all_predictions = []
        weights = []

        # 树模型预测
        tree_models = ['lgb', 'xgb', 'cat', 'et', 'rf']
        neural_models = ['transformer', 'cnn', 'lstm']

        for model_name in tree_models:
            if model_name in models:
                try:
                    model_info = models[model_name]
                    pred_proba = model_info['model'].predict_proba(X_test_clean)
                    all_predictions.append(pred_proba)
                    weight = model_info['score']
                    weights.append(weight)
                    print(f"{model_name}预测完成，权重: {weight:.4f}")
                except Exception as e:
                    print(f"{model_name}预测失败: {e}")

        # 神经网络预测
        for model_name in neural_models:
            if model_name in models:
                try:
                    model_info = models[model_name]
                    scaler = model_info['scaler']
                    X_test_scaled = scaler.transform(X_test_clean)
                    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
                    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)

                    model = model_info['model']
                    model.eval()

                    with torch.no_grad():
                        outputs = model(X_test_tensor)
                        probs = torch.softmax(outputs, dim=1).cpu().numpy()

                    all_predictions.append(probs)
                    weights.append(model_info['score'])
                    print(f"{model_name}预测完成，权重: {model_info['score']:.4f}")
                except Exception as e:
                    print(f"{model_name}预测失败: {e}")

        # 归一化权重
        weights = [weighti * weighti for weighti in weights]
        weights = np.array(weights)
        weights = weights / weights.sum()

        print(f"最终权重分布: {weights}")

        # 加权平均
        ensemble_pred_proba = np.zeros_like(all_predictions[0])
        for pred, weight in zip(all_predictions, weights):
            ensemble_pred_proba += pred * weight

        # 获取最终预测
        final_predictions = np.argmax(ensemble_pred_proba, axis=1)

        # 返回概率和预测（修改点）
        return final_predictions, ensemble_pred_proba

    def post_process_predictions(self, predictions, probabilities, threshold=0.3, target_ratio=0.15):
        """
        后处理：增加label=1的预测数量

        参数:
        - predictions: 原始预测结果
        - probabilities: 预测概率 (n_samples, 3)
        - threshold: label=1概率的阈值，超过此值的样本更容易被改为1
        - target_ratio: 期望label=1占总样本的比例
        """
        print("\n开始后处理...")
        predictions = predictions.copy()

        # 统计原始预测分布
        original_counts = pd.Series(predictions).value_counts().sort_index()
        print(f"原始预测分布:\n{original_counts}")

        # 找出预测为0或2，但label=1概率较高的样本
        label_1_prob = probabilities[:, 1]  # label=1的概率

        # 找出预测为0或2的样本索引
        candidates = np.where((predictions == 0) | (predictions == 2))[0]

        # 按照label=1的概率排序
        candidates_sorted = candidates[np.argsort(-label_1_prob[candidates])]

        # 计算需要转换多少样本为label=1
        current_label1_count = np.sum(predictions == 1)
        target_label1_count = int(len(predictions) * target_ratio)
        needed_conversions = max(0, target_label1_count - current_label1_count)

        print(f"当前label=1数量: {current_label1_count}")
        print(f"目标label=1数量: {target_label1_count}")
        print(f"需要转换: {needed_conversions}个样本")

        # 从候选样本中选择概率最高且超过阈值的样本
        converted_count = 0
        for idx in candidates_sorted:
            if converted_count >= needed_conversions:
                break

            # 检查label=1的概率是否超过阈值
            if label_1_prob[idx] >= threshold:
                predictions[idx] = 1
                converted_count += 1

        print(f"实际转换了 {converted_count} 个样本")

        # 统计后处理后的预测分布
        final_counts = pd.Series(predictions).value_counts().sort_index()
        print(f"后处理后预测分布:\n{final_counts}")

        return predictions

    def train_and_predict(self, train_path, test_path):
        """主训练预测流程"""
        # 加载数据
        print("加载数据...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        print(f"训练数据形状: {train_df.shape}")
        print(f"测试数据形状: {test_df.shape}")

        # 特征工程
        print("\n进行超级特征工程...")
        train_fe = self.create_ultra_features(train_df)
        test_fe = self.create_ultra_features(test_df)

        # 选择特征
        non_feature_cols = ['WELL', 'DEPTH', 'label', 'ID', 'id']
        self.feature_columns = [col for col in train_fe.columns
                                if col not in non_feature_cols
                                and col in test_fe.columns
                                and train_fe[col].dtype in ['float64', 'int64']]

        print(f"最终特征数量: {len(self.feature_columns)}")

        # 准备数据
        X_train = train_fe[self.feature_columns]
        y_train = train_fe['label']
        X_test = test_fe[self.feature_columns]

        # 数据清理
        X_train = self.robust_clean_data(X_train)
        X_test = self.robust_clean_data(X_test)

        # 划分训练集和验证集
        if 'WELL' in train_fe.columns:
            unique_wells = train_fe['WELL'].unique()
            np.random.shuffle(unique_wells)
            split_idx = int(0.8 * len(unique_wells))
            train_wells = unique_wells[:split_idx]
            val_wells = unique_wells[split_idx:]

            train_mask = train_fe['WELL'].isin(train_wells)
            val_mask = train_fe['WELL'].isin(val_wells)

            X_train_split = X_train[train_mask]
            y_train_split = y_train[train_mask]
            X_val_split = X_train[val_mask]
            y_val_split = y_train[val_mask]
        else:
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )

        print(f"训练集: {X_train_split.shape}, 验证集: {X_val_split.shape}")

        # 训练所有模型
        all_models = {}

        # 训练树模型
        tree_models = self.train_tree_models(X_train_split, y_train_split, X_val_split, y_val_split)
        all_models.update(tree_models)

        # 训练神经网络模型
        neural_models = self.train_neural_models(X_train_split, y_train_split, X_val_split, y_val_split)
        all_models.update(neural_models)

        # 集成预测（修改：接收概率）
        test_predictions, test_probabilities = self.weighted_average_ensemble(all_models, X_test)

        # 后处理：调整label=1的数量（新增）
        test_predictions_postprocessed = self.post_process_predictions(
            test_predictions,
            test_probabilities,
            threshold=0.45,  # 可以调整这个阈值
            target_ratio=0.15  # 可以调整目标比例
        )

        # 创建提交文件
        id_column = 'ID' if 'ID' in test_df.columns else 'id'
        submission_ids = test_df[id_column].values if id_column in test_df.columns else range(
            len(test_predictions_postprocessed))

        submission = pd.DataFrame({
            'id': submission_ids,
            'predict': test_predictions_postprocessed
        })

        return submission


# 主程序
if __name__ == "__main__":
    identifier = UltraFastLithologyIdentifier()

    try:
        submission = identifier.train_and_predict(
            "train.csv",
            "validation_without_label.csv"
        )

        if submission is not None:
            submission.to_csv("submission_ultra_fast_0.73.csv", index=False)
            print("\n预测完成！结果已保存到 submission_ultra_fast_0.73.csv")
            print(f"提交文件形状: {submission.shape}")
            print(f"预测值分布:\n{submission['predict'].value_counts().sort_index()}")
        else:
            print("训练失败，无法生成预测结果")

    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback

        traceback.print_exc()
# 下面是训练日志：
# Using device: cuda
# 加载数据...
# 训练数据形状: (38225, 7)
# 测试数据形状: (7086, 6)
# 进行超级特征工程...
# 创建核心特征...
# 创建滑窗特征...
# 创建差分特征...
# 创建地质物理特征...
# 创建交叉特征...
# 特征工程完成，进行最终清理...
# 特征工程完成，总特征数: 887
# 创建核心特征...
# 创建滑窗特征...
# 创建差分特征...
# 创建地质物理特征...
# 创建交叉特征...
# 特征工程完成，进行最终清理...
# 特征工程完成，总特征数: 887
# 最终特征数量: 887
# 训练集: (32474, 887), 验证集: (5751, 887)
# 训练树模型...
# 训练LightGBM...
# Training until validation scores don't improve for 100 rounds
# Early stopping, best iteration is:
# [104]	valid_0's multi_logloss: 0.577062
# LightGBM验证F1: 0.5560
# 训练XGBoost...
# XGBoost验证F1: 0.5685
# 训练CatBoost...
# CatBoost验证F1: 0.5726
# 训练Extra Trees...
# Extra Trees验证F1: 0.9625
# 训练Random Forest...
# Random Forest验证F1: 0.9629
# 训练神经网络模型...
# 训练 transformer...
# transformer验证F1: 0.6046
# 训练 cnn...
# cnn验证F1: 0.5486
# 训练 lstm...
# lstm验证F1: 0.6273
# 进行加权平均集成预测...
# lgb预测完成，权重: 0.5560
# xgb预测完成，权重: 0.5685
# cat预测完成，权重: 0.5726
# et预测完成，权重: 0.9625
# rf预测完成，权重: 0.9629
# transformer预测完成，权重: 0.6046
# cnn预测完成，权重: 0.5486
# lstm预测完成，权重: 0.6273
# 最终权重分布: [0.0797952  0.08344056 0.08463096 0.23914838 0.23932254 0.09437574
#  0.07769703 0.10158958]
# 开始后处理...
# 原始预测分布:
# 0     747
# 1     914
# 2    5425
# Name: count, dtype: int64
# 当前label=1数量: 914
# 目标label=1数量: 1062
# 需要转换: 148个样本
# 实际转换了 30 个样本
# 后处理后预测分布:
# 0     729
# 1     944
# 2    5413
# Name: count, dtype: int64
# 预测完成！结果已保存到 submission_ultra_fast.csv
# 提交文件形状: (7086, 2)
# 预测值分布:
# predict
# 0     729
# 1     944
# 2    5413
# Name: count, dtype: int64
# Click to add a cell.
# 后续优化方向：
# 1.细调参数，比如将catboost的max_depth也调大点，我才调为8，主要是调大训练速度太慢，我等不了；2.可以试试tcn模型；3.可以尝试一下别的损失函数；4.现在的权重是各个验证集分数的平方归一化加权作为权重，可以试试三次方后归一化；5.et模型和rf模型的验证集分数有点虚高，可以试着把他们的权重调小一点
