# ——— 中国石化 AI 大赛：基于测井曲线的岩性识别与分类 ———
# 该文件是在用户提供的 baseline 基础上，逐行添加中文注释，
# 方便理解整体数据流、特征工程、模型训练与集成预测流程。

import pandas as pd  # 数据处理
import numpy as np  # 数值计算
from sklearn.model_selection import GroupKFold, StratifiedKFold  # 分组/分层交叉验证
from sklearn.preprocessing import StandardScaler, RobustScaler  # 特征缩放器
from sklearn.metrics import f1_score  # 评估指标（宏平均 F1）
import lightgbm as lgb  # LightGBM 分类器
import xgboost as xgb  # XGBoost 分类器
from catboost import CatBoostClassifier  # CatBoost 分类器
import torch  # PyTorch 主库
import torch.nn as nn  # 神经网络层
import torch.optim as optim  # 优化器
from torch.utils.data import Dataset, DataLoader  # 数据集与数据加载
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts  # 学习率调度器
import warnings  # 警告控制
warnings.filterwarnings('ignore')  # 忽略非致命警告，输出更干净

# =========================
# 通用：随机种子
# =========================
# 固定随机种子以保证实验可复现

def set_seed(seed=42):
    np.random.seed(seed)  # Numpy 随机数
    torch.manual_seed(seed)  # CPU 上的 torch 随机数
    torch.cuda.manual_seed(seed)  # 单 GPU 随机种子
    torch.cuda.manual_seed_all(seed)  # 多 GPU 随机种子
    torch.backends.cudnn.deterministic = True  # CUDNN 设为确定性
    torch.backends.cudnn.benchmark = False  # 关闭 benchmark 提升确定性

set_seed(42)  # 设定默认随机种子

# 选择运行设备：GPU 优先
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =========================
# 数据集封装（供 PyTorch 使用）
# =========================
class WellLogDataset(Dataset):
    """测井数据集：接收特征和可选的标签，返回张量。"""
    def __init__(self, features, labels=None):
        self.features = torch.FloatTensor(features)  # 特征转为 FloatTensor
        self.labels = torch.LongTensor(labels) if labels is not None else None  # 标签转为 LongTensor

    def __len__(self):
        return len(self.features)  # 数据集样本数

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]  # 训练/验证：返回 (x, y)
        return self.features[idx]  # 测试：仅返回 x

# =========================
# Transformer 基本块
# =========================
class TransformerBlock(nn.Module):
    """标准 Transformer Encoder Block（自注意力 + 前馈 + 残差 + LayerNorm）。"""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        # 多头自注意力（batch_first=True 以 [B, T, C] 作为输入）
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)  # 注意力后的 LayerNorm
        self.norm2 = nn.LayerNorm(embed_dim)  # 前馈后的 LayerNorm

        # 前馈网络：Linear -> GELU -> Dropout -> Linear -> Dropout
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # 自注意力 + 残差
        attn_out, _ = self.attention(x, x, x)  # 自注意力输出和权重（权重此处未用）
        x = self.norm1(x + attn_out)  # 残差连接后归一化

        # 前馈 + 残差
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        return x

# =========================
# 面向表格数据的 Transformer
# =========================
class TabularTransformer(nn.Module):
    """把一条 tabular 向量当作长度为 1 的序列做编码，然后分类。"""
    def __init__(self, input_dim, num_classes=3, embed_dim=128, num_heads=8,
                 num_layers=4, ff_dim=256, dropout=0.2):
        super().__init__()
        # 将原始特征投影到 embedding 维度
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )
        # 可学习的位置编码（长度 100，足够覆盖极短序列，这里实际只用到 1）
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, embed_dim))
        # 堆叠多个 TransformerBlock
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        # 分类头：降维 -> GELU -> Dropout -> 输出类别
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)  # 批大小（未直接使用）
        x = self.input_projection(x)  # 线性投影到 embed_dim
        x = x.unsqueeze(1)  # 增加时间维度，形状变为 [B, 1, C]
        x = x + self.positional_encoding[:, :x.size(1), :]  # 加上位置编码
        for transformer in self.transformer_blocks:  # 通过多层 Transformer
            x = transformer(x)
        x = x.mean(dim=1)  # 全局平均池化到 [B, C]
        output = self.classifier(x)  # 分类输出 [B, num_classes]
        return output

# =========================
# 1D-CNN 模型
# =========================
class CNN1DModel(nn.Module):
    """将特征向量视作长度为 input_dim 的一维信号做卷积特征提取，再分类。"""
    def __init__(self, input_dim, num_classes=3, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),  # conv 块 1
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),  # conv 块 2
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 下采样一半
            nn.Dropout(dropout),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),  # conv 块 3
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # 自适应池化到长度 1
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, hidden_dim),  # 展平后接全连接
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, F] -> [B, 1, F]
        x = self.conv_blocks(x)  # 卷积堆叠
        x = x.squeeze(-1)  # [B, C, 1] -> [B, C]
        output = self.classifier(x)  # 分类
        return output

# =========================
# CNN + Transformer 融合（本 baseline 中未使用该类进行训练）
# =========================
class HybridModel(nn.Module):
    """混合模型：并联 CNN 与 Transformer，拼接后再分类。"""
    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        self.cnn = CNN1DModel(input_dim, num_classes, hidden_dim=128)  # CNN 分支
        self.transformer = TabularTransformer(  # Transformer 分支
            input_dim, num_classes, embed_dim=128, num_heads=8, num_layers=3, ff_dim=256
        )
        self.fusion = nn.Sequential(  # 融合后再分类
            nn.Linear(num_classes * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        cnn_out = self.cnn(x)  # CNN 输出 logits
        transformer_out = self.transformer(x)  # Transformer 输出 logits
        combined = torch.cat([cnn_out, transformer_out], dim=1)  # 拼接
        output = self.fusion(combined)  # 融合分类
        return output

# =========================
# 主流程封装类
# =========================
class AdvancedLithologyIdentifier:
    """高级岩性识别系统：特征工程 + 多模型训练 + 集成预测。"""

    def __init__(self):
        self.models = {}  # 预留：存放模型
        self.feature_columns = []  # 记录用于训练/预测的特征列
        self.scaler = RobustScaler()  # 神经网络部分使用的鲁棒缩放器（对异常值更稳定）

    # ---------- 工具函数 ----------
    def safe_divide(self, a, b, default=0):
        """安全除法：避免 0 除、NaN/Inf，返回替代值。"""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.divide(a, b)
            result = np.where(np.isfinite(result), result, default)
        return result

    def robust_clean_data(self, df):
        """鲁棒数据清理：去除 Inf，裁剪极值，填充缺失。"""
        df = df.copy()
        df = df.replace([np.inf, -np.inf], np.nan)  # 先将无穷置为 NaN
        numeric_cols = df.select_dtypes(include=[np.number]).columns  # 仅处理数值列
        for col in numeric_cols:
            if col not in ['WELL', 'DEPTH', 'label', 'ID', 'id']:
                Q1 = df[col].quantile(0.05)  # 下分位数（更严格）
                Q3 = df[col].quantile(0.95)  # 上分位数（更严格）
                IQR = Q3 - Q1
                lower_bound = Q1 - 2.5 * IQR  # 下界
                upper_bound = Q3 + 2.5 * IQR  # 上界
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)  # 裁剪极端值
                median_val = df[col].median()  # 用中位数更稳健
                if not np.isfinite(median_val):
                    median_val = 0
                df[col] = df[col].fillna(median_val)  # 填充缺失
                df[col] = df[col].replace([np.inf, -np.inf], median_val)  # 再次兜底
        return df

    def clean_data(self, df):
        """对外暴露的清理函数（当前直接调用鲁棒版本）。"""
        return self.robust_clean_data(df)

    # ---------- 特征工程 ----------
    def create_advanced_features(self, df):
        """构造深度、滑窗、地质物理派生、交互、变换、rank、井内标准化等特征。"""
        df = df.copy()
        basic_features = ['SP', 'GR', 'AC']  # 基本测井曲线

        # 深度相关特征：归一化、差分、一阶差分的差分
        df['DEPTH_normalized'] = df.groupby('WELL')['DEPTH'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
        )
        df['DEPTH_diff'] = df.groupby('WELL')['DEPTH'].diff().fillna(0)
        df['DEPTH_diff2'] = df.groupby('WELL')['DEPTH_diff'].diff().fillna(0)

        # 多尺度滑动窗口：均值/标准差（中心对齐），以及窗口差分
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

        # 地质物理启发特征：黏土含量近似 Vsh、声波孔隙度近似等
        if 'GR' in df.columns:
            df['GR_min'] = df.groupby('WELL')['GR'].transform('min')
            df['GR_max'] = df.groupby('WELL')['GR'].transform('max')
            df['Vsh_linear'] = self.safe_divide(df['GR'] - df['GR_min'], df['GR_max'] - df['GR_min'] + 1e-8)
            df['Vsh_linear'] = np.clip(df['Vsh_linear'], 0, 1)

        if 'AC' in df.columns:
            df['PHI_AC'] = self.safe_divide(df['AC'] - 180, 300 - 180)  # 简化的 AC→孔隙度估计
            df['PHI_AC'] = np.clip(df['PHI_AC'], 0, 1)

        if 'Vsh_linear' in df.columns:
            df['PHI_density_proxy'] = 1 - df['Vsh_linear']  # 密度孔隙度代理（启发式）

        # 特征交互：比值类
        if all(col in df.columns for col in ['GR', 'AC']):
            df['GR_AC_ratio'] = self.safe_divide(df['GR'], df['AC'] + 1e-8)
        if all(col in df.columns for col in ['SP', 'GR']):
            df['SP_GR_ratio'] = self.safe_divide(df['SP'], df['GR'] + 1e-8)
        if all(col in df.columns for col in ['SP', 'AC']):
            df['SP_AC_ratio'] = self.safe_divide(df['SP'], df['AC'] + 1e-8)

        # 单调变换：log1p / sqrt（统一 shift 防止负数）
        for feature in basic_features:
            if feature in df.columns:
                df[f'{feature}_log'] = np.log1p(np.maximum(0, df[feature] - df[feature].min() + 1e-8))
                df[f'{feature}_sqrt'] = np.sqrt(np.maximum(0, df[feature] - df[feature].min() + 1e-8))

        # 排名特征：井内分位排名（0-1）
        for feature in basic_features:
            if feature in df.columns:
                df[f'{feature}_rank'] = df.groupby('WELL')[feature].rank(pct=True)

        # 井内标准化：减均值除标准差（safe_divide 兜底）
        for feature in basic_features:
            if feature in df.columns:
                df[f'{feature}_well_normalized'] = df.groupby('WELL')[feature].transform(
                    lambda x: self.safe_divide(x - x.mean(), x.std() + 1e-8)
                )

        # 清理临时列
        df = df.drop(['GR_min', 'GR_max'], axis=1, errors='ignore')
        # 末尾再做一次鲁棒清理
        df = self.robust_clean_data(df)
        return df

    # ---------- 神经网络训练 ----------
    def train_neural_models(self, X_train, y_train, X_val, y_val):
        """训练 Transformer 与 1D-CNN 两个子模型，并记录各自最佳验证 F1。"""
        print("\n训练神经网络模型...")
        # 先做缩放（RobustScaler 对异常值稳定）
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        # 将所有无效值替换为 0（防止数值问题）
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        # 构建数据加载器
        train_dataset = WellLogDataset(X_train_scaled, y_train.values)
        val_dataset = WellLogDataset(X_val_scaled, y_val.values)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
        input_dim = X_train_scaled.shape[1]  # 特征维度
        # 待训练的模型配置列表
        models_config = [
            ('transformer', TabularTransformer(input_dim, num_classes=3, embed_dim=128,
                                              num_heads=8, num_layers=3, ff_dim=256)),
            ('cnn', CNN1DModel(input_dim, num_classes=3, hidden_dim=128)),
        ]
        best_models = {}  # 存放最佳状态的模型
        for model_name, model_class in models_config:
            print(f"\n训练 {model_name} 模型...")
            model = model_class.to(device)  # 放到设备
            criterion = nn.CrossEntropyLoss()  # 多分类交叉熵
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # AdamW
            total_steps = len(train_loader) * 30  # 预估 30 个 epoch 的总步数
            scheduler = OneCycleLR(
                optimizer, max_lr=0.01, total_steps=total_steps, pct_start=0.3, anneal_strategy='cos'
            )  # OneCycle 学习率策略
            best_val_f1 = 0  # 记录最佳验证 F1
            best_model_state = None  # 记录最佳权重
            patience_counter = 0  # 早停计数
            patience = 8  # 容忍轮次
            for epoch in range(30):  # 最多 30 轮
                model.train()
                train_loss = 0
                train_preds, train_labels = [], []
                for batch_features, batch_labels in train_loader:
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
                    optimizer.step()
                    scheduler.step()  # 更新学习率
                    train_loss += loss.item()
                    train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                    train_labels.extend(batch_labels.cpu().numpy())
                # 验证
                model.eval()
                val_preds, val_labels = [], []
                with torch.no_grad():
                    for batch_features, batch_labels in val_loader:
                        batch_features = batch_features.to(device)
                        outputs = model(batch_features)
                        val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                        val_labels.extend(batch_labels.cpu().numpy())
                train_f1 = f1_score(train_labels, train_preds, average='macro')
                val_f1 = f1_score(val_labels, val_preds, average='macro')
                if epoch % 5 == 0:
                    print(f"Epoch {epoch}: Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
                # 保存最佳
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:  # 早停
                    print(f"Early stopping at epoch {epoch}")
                    break
            if best_model_state is not None:  # 载入最佳
                model.load_state_dict(best_model_state)
            best_models[model_name] = {'model': model, 'cv_score': best_val_f1, 'scaler': self.scaler}
            print(f"{model_name} 最佳验证F1: {best_val_f1:.4f}")
        return best_models

    # ---------- 树模型训练（LGB/XGB/CAT） ----------
    def train_tree_models_advanced(self, X_train, y_train, groups):
        """对树模型做更稳健的数据清洗与交叉验证训练。"""
        print("\n训练高级树模型...")
        X_train_clean = X_train.copy().replace([np.inf, -np.inf], np.nan)  # 初步清理
        # 逐列填充 & 兜底
        for col in X_train_clean.columns:
            col_data = X_train_clean[col]
            median_val = col_data.median()
            if not np.isfinite(median_val):
                median_val = 0
            X_train_clean[col] = col_data.fillna(median_val)
            X_train_clean[col] = X_train_clean[col].replace([np.inf, -np.inf], median_val)
            if np.any(~np.isfinite(X_train_clean[col].values)):
                print(f"警告: 列 {col} 仍包含无效值，使用0填充")
                X_train_clean[col] = X_train_clean[col].replace([np.inf, -np.inf, np.nan], 0)
        if np.any(~np.isfinite(X_train_clean.values)):
            print("最终清理：替换所有无效值为0")
            X_train_clean = pd.DataFrame(
                np.nan_to_num(X_train_clean.values, nan=0.0, posinf=0.0, neginf=0.0),
                columns=X_train_clean.columns, index=X_train_clean.index
            )
        # 各模型参数（较稳健的默认）
        lgb_params = {
            'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss', 'learning_rate': 0.02,
            'num_leaves': 31, 'max_depth': 8, 'min_child_samples': 20, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1, 'random_state': 42,
            'n_estimators': 1000, 'verbose': -1
        }
        xgb_params = {
            'objective': 'multi:softprob', 'num_class': 3, 'learning_rate': 0.02, 'max_depth': 8,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
            'random_state': 42, 'n_estimators': 1000, 'tree_method': 'hist', 'eval_metric': 'mlogloss',
            'verbosity': 0, 'missing': np.nan
        }
        cat_params = {
            'loss_function': 'MultiClass', 'learning_rate': 0.03, 'depth': 6, 'l2_leaf_reg': 5,
            'random_seed': 42, 'iterations': 1000, 'od_type': 'Iter', 'od_wait': 50, 'verbose': False
        }
        models = {}
        n_groups = len(np.unique(groups))  # 井的数量（用于 GroupKFold）
        n_splits = min(5, n_groups)  # 最多 5 折
        if n_splits > 1:
            gkf = GroupKFold(n_splits=n_splits)
            splits = list(gkf.split(X_train_clean, y_train, groups=groups))
        else:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 退化为分层 K 折
            splits = list(skf.split(X_train_clean, y_train))
        # 逐模型训练
        for model_name, params in [('lgb', lgb_params), ('xgb', xgb_params), ('cat', cat_params)]:
            print(f"\n训练 {model_name.upper()} 模型...")
            fold_models, fold_scores = [], []
            for fold, (train_idx, val_idx) in enumerate(splits):
                X_tr, X_val = X_train_clean.iloc[train_idx], X_train_clean.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                X_tr = X_tr.replace([np.inf, -np.inf], np.nan).fillna(X_tr.median())  # 再清理
                X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(X_val.median())
                try:
                    if model_name == 'lgb':
                        model = lgb.LGBMClassifier(**params)
                        model.fit(
                            X_tr, y_tr, eval_set=[(X_val, y_val)],
                            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
                        )
                    elif model_name == 'xgb':
                        model = xgb.XGBClassifier(**params)
                        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                    else:  # catboost
                        model = CatBoostClassifier(**params)
                        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                    fold_models.append(model)
                    val_pred = model.predict(X_val)  # 直接类别预测
                    fold_f1 = f1_score(y_val, val_pred, average='macro')
                    fold_scores.append(fold_f1)
                    print(f"Fold {fold+1} F1: {fold_f1:.4f}")
                except Exception as e:  # 某折失败则回退到 RF
                    print(f"Fold {fold+1} 训练失败: {e}")
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_tr, y_tr)
                    fold_models.append(model)
                    val_pred = model.predict(X_val)
                    fold_f1 = f1_score(y_val, val_pred, average='macro')
                    fold_scores.append(fold_f1)
                    print(f"Fold {fold+1} 使用备选模型 F1: {fold_f1:.4f}")
            models[model_name] = {'models': fold_models, 'cv_score': np.mean(fold_scores)}
            print(f"{model_name.UPPER()} 平均CV分数: {np.mean(fold_scores):.4f}")
        return models

    # ---------- 集成推理 ----------
    def ensemble_predictions_advanced(self, tree_models, neural_models, X_test):
        """对树模型（多折）与神经网络结果做加权融合，权重来自各自验证分数。"""
        print("\n进行高级集成预测...")
        X_test_clean = X_test.copy().replace([np.inf, -np.inf], np.nan)  # 初步清理
        # 逐列兜底清理
        for col in X_test_clean.columns:
            col_data = X_test_clean[col]
            median_val = col_data.median()
            if not np.isfinite(median_val):
                median_val = 0
            X_test_clean[col] = col_data.fillna(median_val)
            X_test_clean[col] = X_test_clean[col].replace([np.inf, -np.inf], median_val)
        if np.any(~np.isfinite(X_test_clean.values)):
            print("清理测试数据中的异常值...")
            X_test_clean = pd.DataFrame(
                np.nan_to_num(X_test_clean.values, nan=0.0, posinf=0.0, neginf=0.0),
                columns=X_test_clean.columns, index=X_test_clean.index
            )
        all_predictions, weights = [], []  # 保存各模型概率与权重
        # 树模型预测（对每折取均值概率）
        for model_name, model_info in tree_models.items():
            fold_preds = []
            for model in model_info['models']:
                try:
                    pred = model.predict_proba(X_test_clean)  # 概率输出 [N, 3]
                    fold_preds.append(pred)
                except Exception as e:
                    print(f"{model_name} 预测失败: {e}")
                    pred = np.ones((len(X_test_clean), 3)) / 3  # 退化为均匀概率
                    fold_preds.append(pred)
            avg_pred = np.mean(fold_preds, axis=0)  # 折内平均
            all_predictions.append(avg_pred)
            weights.append(model_info['cv_score'])  # 权重 = 该模型的 CV 分
        # 神经网络预测（先用与之匹配的 scaler 进行缩放）
        if neural_models:
            X_test_scaled = list(neural_models.values())[0]['scaler'].transform(X_test_clean)
            X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
            for model_name, model_info in neural_models.items():
                model = model_info['model']
                model.eval()
                with torch.no_grad():
                    outputs = model(X_test_tensor)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()  # 概率化
                all_predictions.append(probs)
                weights.append(model_info['cv_score'])
        # 归一化权重
        weights = np.array(weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(weights)) / len(weights)
        print(f"模型权重: {weights}")
        # 加权求和得到最终概率
        ensemble_pred = np.zeros_like(all_predictions[0])
        for pred, weight in zip(all_predictions, weights):
            ensemble_pred += pred * weight
        final_predictions = np.argmax(ensemble_pred, axis=1)  # 取概率最大类别
        return final_predictions

    # ---------- 总控：训练 + 预测 ----------
    def train_and_predict(self, train_path, test_path):
        """完整流程：读取数据 → 特征工程 → 交叉验证训练 → 集成预测 → 生成提交。"""
        print("加载数据...")
        train_df = pd.read_csv(train_path)  # 训练集
        test_df = pd.read_csv(test_path)  # 测试集（无标签）
        print(f"训练数据形状: {train_df.shape}")
        print(f"测试数据形状: {test_df.shape}")
        if 'label' in train_df.columns:
            print(f"训练数据岩性分布:\n{train_df['label'].value_counts().sort_index()}")
        print("\n进行特征工程...")
        train_fe = self.create_advanced_features(train_df)  # 训练集派生
        test_fe = self.create_advanced_features(test_df)  # 测试集派生
        # 过滤出双方都存在且为数值型的特征
        non_feature_cols = ['WELL', 'DEPTH', 'label', 'ID', 'id']
        self.feature_columns = [
            col for col in train_fe.columns
            if col not in non_feature_cols and col in test_fe.columns and train_fe[col].dtype in ['float64', 'int64']
        ]
        print(f"特征数量: {len(self.feature_columns)}")
        X_train = train_fe[self.feature_columns]  # 训练特征
        y_train = train_fe['label'] if 'label' in train_fe.columns else None  # 标签
        groups = train_fe['WELL'] if 'WELL' in train_fe.columns else None  # 分组依据（井）
        X_test = test_fe[self.feature_columns]  # 测试特征
        print(f"训练数据特征形状: {X_train.shape}")
        print("进行最终数据清理...")
        X_train = self.robust_clean_data(X_train)
        X_test = self.robust_clean_data(X_test)
        if y_train is None:  # 若无标签则无法继续
            print("错误: 训练数据中没有找到label列")
            return None
        # 构造一个验证切分：优先用 GroupKFold（按井不泄漏），否则简单切分
        if groups is not None and len(np.unique(groups)) > 1:
            n_groups = len(np.unique(groups))
            gkf = GroupKFold(n_splits=min(5, n_groups))
            splits = list(gkf.split(X_train, y_train, groups=groups))
            if splits:
                train_idx, val_idx = splits[0]
            else:
                val_size = int(0.2 * len(X_train))
                train_idx = list(range(len(X_train) - val_size))
                val_idx = list(range(len(X_train) - val_size, len(X_train)))
        else:
            val_size = int(0.2 * len(X_train))
            train_idx = list(range(len(X_train) - val_size))
            val_idx = list(range(len(X_train) - val_size, len(X_train)))
        X_train_nn, X_val_nn = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_nn, y_val_nn = y_train.iloc[train_idx], y_train.iloc[val_idx]
        # 训练树模型（全量 X_train / y_train / groups）
        tree_models = self.train_tree_models_advanced(X_train, y_train, groups)
        # 训练神经网络（基于一次切分的 train/val）
        neural_models = self.train_neural_models(X_train_nn, y_train_nn, X_val_nn, y_val_nn)
        # 集成预测
        test_predictions = self.ensemble_predictions_advanced(tree_models, neural_models, X_test)
        # 生成提交（优先使用 id 列）
        id_column = None
        for col in ['ID', 'id']:
            if col in test_df.columns:
                id_column = col
                break
        if id_column:
            submission_ids = test_df[id_column].values
        else:
            submission_ids = range(len(test_predictions))
        submission = pd.DataFrame({'id': submission_ids, 'predict': test_predictions})
        return submission

# =========================
# 脚本入口
# =========================
if __name__ == "__main__":
    identifier = AdvancedLithologyIdentifier()  # 实例化主流程类
    try:
        submission = identifier.train_and_predict(
            "SINOPEC-01/train.csv",  # 训练数据相对路径
            "SINOPEC-01/validation_without_label.csv"  # 测试数据相对路径
        )
        if submission is not None:
            submission.to_csv("submission_advanced.csv", index=False)  # 保存提交文件
            print("\n预测完成！结果已保存到 submission_advanced.csv")
            print(f"提交文件形状: {submission.shape}")
            print(f"预测值分布:\n{submission['predict'].value_counts().sort_index()}")
        else:
            print("训练失败，无法生成预测结果")
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()
