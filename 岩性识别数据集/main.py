# 中国石化人工智能大赛：基于测井曲线的岩性识别与分类算法0.62829的Baseline方案
# 比赛的地址：
# https://aicup.sinopec.com/competition/SINOPEC-01/

# 背景介绍
# 在油气勘探与开发领域，地层岩性（即岩石类型）的精确识别是资源评价、储层预测和钻井工程设计的基础。测井技术通过在钻孔中下放精密仪器，连续测量地层沿井深的各种地球物理参数（如电学、声学、放射性等），形成了丰富的测井曲线数据。这些曲线蕴含着关于地下岩石类型、物性（孔隙度、渗透率）及流体性质（油、气、水）的关键信息。
 
# 传统上，岩性识别主要依赖于地质工程师的人工解释。该方法不仅耗时耗力，而且解释结果高度依赖于个人经验，难以实现标准化和规模化。随着人工智能技术的发展，利用机器学习算法分析多维度的测井曲线，能够自动、高效、精准地识别岩性，已经成为地球物理勘探领域的研究热点。
 
# 本次竞赛旨在鼓励参赛者探索和应用前沿的人工智能技术，通过分析真实的测井数据，构建一个高精度的岩性自动识别模型。这不仅能极大提升地质研究的效率和准确性，还能为油气资源的智能化勘探与开发提供坚实的技术支撑，具有重要的学术价值和广阔的工业应用前景。

# 竞赛任务
# 选手需要利用提供的测井曲线数据集，构建一个或多个高精度的机器学习或深度学习模型，对给定深度点的地层岩性进行自动识别和分类。具体任务是将每个深度点的岩性准确地分类为砂岩、粉砂岩、泥岩三类之一。



# 下面是0.62829分的完整baseline代码：
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import f1_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
import warnings
warnings.filterwarnings('ignore')

print(torch.__version__)
print(lgb.__version__)

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') windows
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

class TransformerBlock(nn.Module):
    """Transformer块"""
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
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)

        return x

class TabularTransformer(nn.Module):
    """用于表格数据的Transformer模型"""
    def __init__(self, input_dim, num_classes=3, embed_dim=128, num_heads=8, 
                 num_layers=4, ff_dim=256, dropout=0.2):
        super().__init__()

        # Feature embedding
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )

        # Positional encoding (learnable)
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, embed_dim))

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # Project input features
        x = self.input_projection(x)
        x = x.unsqueeze(1)  # Add sequence dimension

        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :]

        # Pass through transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x)

        # Global pooling
        x = x.mean(dim=1)

        # Classification
        output = self.classifier(x)

        return output

class CNN1DModel(nn.Module):
    """1D CNN模型用于序列特征提取"""
    def __init__(self, input_dim, num_classes=3, hidden_dim=128, dropout=0.3):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Block 2
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # Reshape for Conv1d: (batch, 1, features)
        x = x.unsqueeze(1)

        # Conv blocks
        x = self.conv_blocks(x)

        # Flatten
        x = x.squeeze(-1)

        # Classification
        output = self.classifier(x)

        return output

class HybridModel(nn.Module):
    """混合模型：结合CNN和Transformer"""
    def __init__(self, input_dim, num_classes=3):
        super().__init__()

        # CNN branch
        self.cnn = CNN1DModel(input_dim, num_classes, hidden_dim=128)

        # Transformer branch
        self.transformer = TabularTransformer(
            input_dim, num_classes, 
            embed_dim=128, num_heads=8, 
            num_layers=3, ff_dim=256
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(num_classes * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        cnn_out = self.cnn(x)
        transformer_out = self.transformer(x)

        # Concatenate outputs
        combined = torch.cat([cnn_out, transformer_out], dim=1)

        # Final prediction
        output = self.fusion(combined)

        return output

class AdvancedLithologyIdentifier:
    """高级岩性识别系统"""

    def __init__(self):
        self.models = {}
        self.feature_columns = []
        self.scaler = RobustScaler()

    def safe_divide(self, a, b, default=0):
        """安全除法，避免除零和无穷值"""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.divide(a, b)
            result = np.where(np.isfinite(result), result, default)
        return result

    def robust_clean_data(self, df):
        """更鲁棒的数据清理方法"""
        df = df.copy()

        # 替换无穷值
        df = df.replace([np.inf, -np.inf], np.nan)

        # 对每个数值列进行异常值处理
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['WELL', 'DEPTH', 'label', 'ID', 'id']:
                # 使用更严格的IQR方法处理异常值
                Q1 = df[col].quantile(0.05)  # 使用5%分位数代替1%
                Q3 = df[col].quantile(0.95)  # 使用95%分位数代替99%
                IQR = Q3 - Q1

                # 更严格的边界
                lower_bound = Q1 - 2.5 * IQR
                upper_bound = Q3 + 2.5 * IQR

                # 裁剪极端值
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

                # 使用更鲁棒的填充方法
                median_val = df[col].median()
                if not np.isfinite(median_val):
                    median_val = 0

                # 填充剩余的NaN值
                df[col] = df[col].fillna(median_val)

                # 最终检查：确保没有无穷值
                df[col] = df[col].replace([np.inf, -np.inf], median_val)

        return df

    def clean_data(self, df):
        """清理数据中的异常值 - 使用更鲁棒的版本"""
        return self.robust_clean_data(df)
    
    def balance_samples(self, X_train, y_train):
        """使用SMOTE平衡样本（可选）"""
        try:
            from imblearn.over_sampling import SMOTE
            print("\n原始样本分布:", y_train.value_counts().to_dict())
            
            smote = SMOTE(random_state=42, k_neighbors=5)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            
            print("平衡后样本分布:", pd.Series(y_resampled).value_counts().to_dict())
            return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled)
        except ImportError:
            print("警告: imblearn未安装，跳过样本平衡")
            return X_train, y_train
        except Exception as e:
            print(f"样本平衡失败: {e}，使用原始数据")
            return X_train, y_train

    def create_advanced_features(self, df):
        """创建高级特征（增强版）"""
        df = df.copy()

        basic_features = ['SP', 'GR', 'AC']

        # 1. 深度相关特征（增强）
        df['DEPTH_normalized'] = df.groupby('WELL')['DEPTH'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
        )
        df['DEPTH_diff'] = df.groupby('WELL')['DEPTH'].diff().fillna(0)
        df['DEPTH_diff2'] = df.groupby('WELL')['DEPTH_diff'].diff().fillna(0)
        
        # 新增：深度分层特征
        df['DEPTH_zone'] = pd.cut(df['DEPTH'], bins=10, labels=False)
        df['DEPTH_zone_normalized'] = df['DEPTH_zone'] / 9.0

        # 2. 多尺度滑动窗口特征 (使用更小的窗口避免边界问题)
        window_sizes = [3, 5, 7]
        for window in window_sizes:
            for feature in basic_features:
                if feature in df.columns:
                    # 统计特征
                    df[f'{feature}_rolling_mean_{window}'] = df.groupby('WELL')[feature].transform(
                        lambda x: x.rolling(window=window, min_periods=1, center=True).mean()
                    )
                    df[f'{feature}_rolling_std_{window}'] = df.groupby('WELL')[feature].transform(
                        lambda x: x.rolling(window=window, min_periods=1, center=True).std().fillna(0)
                    )

                    # 差分特征
                    df[f'{feature}_diff_{window}'] = df.groupby('WELL')[feature].transform(
                        lambda x: x.diff(window).fillna(0)
                    )

        # 3. 地质物理特征（大幅增强）
        if 'GR' in df.columns:
            df['GR_min'] = df.groupby('WELL')['GR'].transform('min')
            df['GR_max'] = df.groupby('WELL')['GR'].transform('max')
            
            # 线性泥质含量
            df['Vsh_linear'] = self.safe_divide(df['GR'] - df['GR_min'], df['GR_max'] - df['GR_min'] + 1e-8)
            df['Vsh_linear'] = np.clip(df['Vsh_linear'], 0, 1)
            
            # 新增：Larionov老地层公式（更准确的泥质含量）
            IGR = df['Vsh_linear']
            df['Vsh_larionov'] = 0.083 * (2**(3.7 * IGR) - 1)
            df['Vsh_larionov'] = np.clip(df['Vsh_larionov'], 0, 1)
            
            # 新增：Steiber公式
            df['Vsh_steiber'] = self.safe_divide(IGR, 3 - 2 * IGR)
            df['Vsh_steiber'] = np.clip(df['Vsh_steiber'], 0, 1)
            
            # 新增：GR归一化（更鲁棒）
            df['GR_normalized'] = self.safe_divide(
                df['GR'] - df.groupby('WELL')['GR'].transform('median'),
                df.groupby('WELL')['GR'].transform('std') + 1e-8
            )

        if 'AC' in df.columns:
            # 孔隙度估算（Wyllie公式）
            df['PHI_AC'] = self.safe_divide(df['AC'] - 180, 300 - 180)
            df['PHI_AC'] = np.clip(df['PHI_AC'], 0, 1)
            
            # 新增：更精确的孔隙度（考虑基质和流体）
            AC_matrix = 182  # 砂岩基质
            AC_fluid = 617   # 流体
            df['PHI_wyllie'] = self.safe_divide(df['AC'] - AC_matrix, AC_fluid - AC_matrix)
            df['PHI_wyllie'] = np.clip(df['PHI_wyllie'], 0, 0.4)
            
            # 新增：声波时差归一化
            df['AC_normalized'] = self.safe_divide(
                df['AC'] - df.groupby('WELL')['AC'].transform('median'),
                df.groupby('WELL')['AC'].transform('std') + 1e-8
            )

        if 'Vsh_linear' in df.columns and 'PHI_AC' in df.columns:
            # 密度孔隙度代理
            df['PHI_density_proxy'] = 1 - df['Vsh_linear']
            
            # 新增：有效孔隙度（去除泥质影响）
            df['PHI_effective'] = df['PHI_AC'] * (1 - df['Vsh_linear'])
            
            # 新增：渗透性指标（Kozeny-Carman近似）
            df['permeability_proxy'] = self.safe_divide(
                df['PHI_effective'] ** 3,
                (1 - df['PHI_effective']) ** 2 + 1e-8
            )
            
        # 新增：SP相关的地质指标
        if 'SP' in df.columns:
            # SP归一化
            df['SP_normalized'] = self.safe_divide(
                df['SP'] - df.groupby('WELL')['SP'].transform('median'),
                df.groupby('WELL')['SP'].transform('std') + 1e-8
            )
            
            # SP基线漂移（相对于井平均值的偏离）
            df['SP_baseline_deviation'] = df['SP'] - df.groupby('WELL')['SP'].transform('mean')

        # 4. 特征交互 - 使用安全除法（增强）
        if all(col in df.columns for col in ['GR', 'AC']):
            df['GR_AC_ratio'] = self.safe_divide(df['GR'], df['AC'] + 1e-8)
            df['GR_AC_product'] = df['GR'] * df['AC']  # 新增：乘积特征
            df['GR_AC_diff'] = df['GR'] - df['AC']     # 新增：差异特征

        if all(col in df.columns for col in ['SP', 'GR']):
            df['SP_GR_ratio'] = self.safe_divide(df['SP'], df['GR'] + 1e-8)
            df['SP_GR_product'] = df['SP'] * df['GR']   # 新增
            df['SP_GR_diff'] = df['SP'] - df['GR']      # 新增

        if all(col in df.columns for col in ['SP', 'AC']):
            df['SP_AC_ratio'] = self.safe_divide(df['SP'], df['AC'] + 1e-8)
            df['SP_AC_product'] = df['SP'] * df['AC']   # 新增
            df['SP_AC_diff'] = df['SP'] - df['AC']      # 新增
            
        # 新增：三元交互特征（岩性判别指标）
        if all(col in df.columns for col in ['SP', 'GR', 'AC']):
            # 砂泥岩判别指数
            df['sand_shale_index'] = self.safe_divide(
                df['SP'] * df['AC'],
                df['GR'] + 1e-8
            )
            
            # 综合岩性指标
            df['lithology_index'] = self.safe_divide(
                df['GR'] * df['AC'],
                np.abs(df['SP']) + 1e-8
            )

        # 5. 对数变换特征 (添加保护避免负数)
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
                
        # 新增：梯度特征（一阶、二阶导数）
        for feature in basic_features:
            if feature in df.columns:
                # 一阶梯度
                df[f'{feature}_gradient'] = df.groupby('WELL')[feature].diff().fillna(0)
                
                # 二阶梯度
                df[f'{feature}_gradient2'] = df.groupby('WELL')[f'{feature}_gradient'].diff().fillna(0)
                
                # 梯度变化率
                df[f'{feature}_gradient_pct'] = df.groupby('WELL')[feature].pct_change().fillna(0)
                
                # 梯度方向（上升/下降/平稳）
                df[f'{feature}_gradient_sign'] = np.sign(df[f'{feature}_gradient'])
        
        # 新增：局部波动性特征
        for feature in basic_features:
            if feature in df.columns:
                # 7点窗口的变异系数
                rolling_mean = df.groupby('WELL')[feature].transform(
                    lambda x: x.rolling(7, min_periods=1, center=True).mean()
                )
                rolling_std = df.groupby('WELL')[feature].transform(
                    lambda x: x.rolling(7, min_periods=1, center=True).std().fillna(0)
                )
                df[f'{feature}_cv'] = self.safe_divide(rolling_std, rolling_mean + 1e-8)
                
                # 局部极值特征（是否为局部最大/最小值）
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
        
        # 新增：频率域特征（简化版FFT特征）
        for feature in basic_features:
            if feature in df.columns:
                # 11点窗口的峰峰值
                df[f'{feature}_peak_to_peak'] = df.groupby('WELL')[feature].transform(
                    lambda x: x.rolling(11, min_periods=1, center=True).apply(
                        lambda y: y.max() - y.min() if len(y) > 0 else 0
                    )
                )

        # 删除临时列
        df = df.drop(['GR_min', 'GR_max'], axis=1, errors='ignore')

        # 最终数据清理
        df = self.robust_clean_data(df)

        return df

    def train_neural_models(self, X_train, y_train, X_val, y_val):
        """训练神经网络模型"""
        print("\n训练神经网络模型...")

        # 数据标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # 再次检查并清理数据
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # 创建数据加载器
        train_dataset = WellLogDataset(X_train_scaled, y_train.values)
        val_dataset = WellLogDataset(X_val_scaled, y_val.values)

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

        input_dim = X_train_scaled.shape[1]

        # 训练多个模型
        models_config = [
            ('transformer', TabularTransformer(input_dim, num_classes=3, embed_dim=128, 
                                              num_heads=8, num_layers=3, ff_dim=256)),
            ('cnn', CNN1DModel(input_dim, num_classes=3, hidden_dim=128)),
        ]

        best_models = {}

        for model_name, model_class in models_config:
            print(f"\n训练 {model_name} 模型...")
            model = model_class.to(device)

            # 损失函数和优化器
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

            # 学习率调度器（增加训练轮数）
            total_steps = len(train_loader) * 50  # 增加到50 epochs
            scheduler = OneCycleLR(
                optimizer, 
                max_lr=0.01,
                total_steps=total_steps,
                pct_start=0.3,
                anneal_strategy='cos'
            )

            best_val_f1 = 0
            best_model_state = None
            patience_counter = 0
            patience = 12  # 增加patience到12

            for epoch in range(50):  # 增加到50轮
                # 训练阶段
                model.train()
                train_loss = 0
                train_preds = []
                train_labels = []

                for batch_features, batch_labels in train_loader:
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()

                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()
                    scheduler.step()

                    train_loss += loss.item()
                    train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                    train_labels.extend(batch_labels.cpu().numpy())

                # 验证阶段
                model.eval()
                val_preds = []
                val_labels = []

                with torch.no_grad():
                    for batch_features, batch_labels in val_loader:
                        batch_features = batch_features.to(device)
                        outputs = model(batch_features)
                        val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                        val_labels.extend(batch_labels.cpu().numpy())

                # 计算F1分数
                train_f1 = f1_score(train_labels, train_preds, average='macro')
                val_f1 = f1_score(val_labels, val_preds, average='macro')

                if epoch % 5 == 0:
                    print(f"Epoch {epoch}: Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")

                # 保存最佳模型
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                # 早停
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            # 加载最佳模型
            if best_model_state is not None:
                model.load_state_dict(best_model_state)

            best_models[model_name] = {
                'model': model,
                'cv_score': best_val_f1,
                'scaler': self.scaler
            }

            print(f"{model_name} 最佳验证F1: {best_val_f1:.4f}")

        return best_models

    def train_tree_models_advanced(self, X_train, y_train, groups):
        """训练高级树模型"""
        print("\n训练高级树模型...")

        # 更严格的数据清理
        X_train_clean = X_train.copy()
        X_train_clean = X_train_clean.replace([np.inf, -np.inf], np.nan)

        # 逐列清理
        for col in X_train_clean.columns:
            col_data = X_train_clean[col]
            # 计算统计量
            median_val = col_data.median()
            if not np.isfinite(median_val):
                median_val = 0

            # 替换无穷值和NaN
            X_train_clean[col] = col_data.fillna(median_val)
            X_train_clean[col] = X_train_clean[col].replace([np.inf, -np.inf], median_val)

            # 最终检查
            if np.any(~np.isfinite(X_train_clean[col].values)):
                print(f"警告: 列 {col} 仍包含无效值，使用0填充")
                X_train_clean[col] = X_train_clean[col].replace([np.inf, -np.inf, np.nan], 0)

        # 检查是否还有无穷值或NaN
        if np.any(~np.isfinite(X_train_clean.values)):
            print("最终清理：替换所有无效值为0")
            X_train_clean = pd.DataFrame(
                np.nan_to_num(X_train_clean.values, nan=0.0, posinf=0.0, neginf=0.0),
                columns=X_train_clean.columns,
                index=X_train_clean.index
            )

        # 参数优化后的模型配置（增加训练轮数，调整早停）
        lgb_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'learning_rate': 0.02,  # 降低学习率以获得更稳定的训练
            'num_leaves': 31,
            'max_depth': 8,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_estimators': 1000,  # 增加到2000轮
            'verbose': -1
        }

        xgb_params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'learning_rate': 0.02,  # 降低学习率
            'max_depth': 8,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_estimators': 1000,  # 增加到2000轮
            'tree_method': 'hist',
            'eval_metric': 'mlogloss',
            'verbosity': 0,
            'missing': np.nan,  # 明确指定缺失值处理
            'early_stopping_rounds': 100  # XGBoost专用早停参数
        }

        cat_params = {
            'loss_function': 'MultiClass',
            'learning_rate': 0.02,  # 稍微降低学习率
            'depth': 6,
            'l2_leaf_reg': 5,
            'random_seed': 42,
            'iterations': 1000,  # 增加到2000轮
            'od_type': 'Iter',
            'od_wait': 100,  # 增加早停patience
            'verbose': False
        }

        models = {}
        n_groups = len(np.unique(groups))
        n_splits = min(5, n_groups)

        if n_splits > 1:
            gkf = GroupKFold(n_splits=n_splits)
            splits = list(gkf.split(X_train_clean, y_train, groups=groups))
        else:
            # 使用StratifiedKFold作为备选
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            splits = list(skf.split(X_train_clean, y_train))

        for model_name, params in [('lgb', lgb_params), ('xgb', xgb_params), ('cat', cat_params)]:
            print(f"\n训练 {model_name.upper()} 模型...")
            fold_models = []
            fold_scores = []

            for fold, (train_idx, val_idx) in enumerate(splits):
                X_tr, X_val = X_train_clean.iloc[train_idx], X_train_clean.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                # 最终数据检查
                X_tr = X_tr.replace([np.inf, -np.inf], np.nan).fillna(X_tr.median())
                X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(X_val.median())

                try:
                    if model_name == 'lgb':
                        model = lgb.LGBMClassifier(**params)
                        model.fit(
                            X_tr, y_tr,
                            eval_set=[(X_val, y_val)],
                            callbacks=[lgb.early_stopping(150), lgb.log_evaluation(50)]  # 增加早停patience，每50轮打印一次
                        )
                    elif model_name == 'xgb':
                        model = xgb.XGBClassifier(**params)
                        model.fit(
                            X_tr, y_tr,
                            eval_set=[(X_val, y_val)],
                            verbose=False
                        )
                    else:  # catboost
                        model = CatBoostClassifier(**params)
                        model.fit(
                            X_tr, y_tr,
                            eval_set=[(X_val, y_val)],
                            verbose=False
                        )

                    fold_models.append(model)

                    # 验证
                    val_pred = model.predict(X_val)
                    fold_f1 = f1_score(y_val, val_pred, average='macro')
                    fold_scores.append(fold_f1)
                    print(f"Fold {fold+1} F1: {fold_f1:.4f}")

                except Exception as e:
                    print(f"Fold {fold+1} 训练失败: {e}")
                    # 使用简单的模型作为备选
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_tr, y_tr)
                    fold_models.append(model)
                    val_pred = model.predict(X_val)
                    fold_f1 = f1_score(y_val, val_pred, average='macro')
                    fold_scores.append(fold_f1)
                    print(f"Fold {fold+1} 使用备选模型 F1: {fold_f1:.4f}")

            models[model_name] = {
                'models': fold_models,
                'cv_score': np.mean(fold_scores)
            }
            print(f"{model_name.upper()} 平均CV分数: {np.mean(fold_scores):.4f}")

        return models

    def ensemble_predictions_advanced(self, tree_models, neural_models, X_test):
        """高级集成预测"""
        print("\n进行高级集成预测...")

        # 清理测试数据
        X_test_clean = X_test.copy()
        X_test_clean = X_test_clean.replace([np.inf, -np.inf], np.nan)

        # 逐列清理
        for col in X_test_clean.columns:
            col_data = X_test_clean[col]
            median_val = col_data.median()
            if not np.isfinite(median_val):
                median_val = 0
            X_test_clean[col] = col_data.fillna(median_val)
            X_test_clean[col] = X_test_clean[col].replace([np.inf, -np.inf], median_val)

        # 最终清理
        if np.any(~np.isfinite(X_test_clean.values)):
            print("清理测试数据中的异常值...")
            X_test_clean = pd.DataFrame(
                np.nan_to_num(X_test_clean.values, nan=0.0, posinf=0.0, neginf=0.0),
                columns=X_test_clean.columns,
                index=X_test_clean.index
            )

        all_predictions = []
        weights = []

        # 树模型预测（多fold平均）
        for model_name, model_info in tree_models.items():
            fold_preds = []
            for model in model_info['models']:
                try:
                    pred = model.predict_proba(X_test_clean)
                    fold_preds.append(pred)
                except Exception as e:
                    print(f"{model_name} 预测失败: {e}")
                    # 使用随机预测作为备选
                    pred = np.ones((len(X_test_clean), 3)) / 3
                    fold_preds.append(pred)

            # 平均多个fold的预测
            avg_pred = np.mean(fold_preds, axis=0)
            all_predictions.append(avg_pred)
            weights.append(model_info['cv_score'])

        # 神经网络预测
        if neural_models:
            X_test_scaled = list(neural_models.values())[0]['scaler'].transform(X_test_clean)
            X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)

            for model_name, model_info in neural_models.items():
                model = model_info['model']
                model.eval()

                with torch.no_grad():
                    outputs = model(X_test_tensor)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()

                all_predictions.append(probs)
                weights.append(model_info['cv_score'])

        # 归一化权重
        weights = np.array(weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(weights)) / len(weights)

        print(f"模型权重: {weights}")

        # 加权集成
        ensemble_pred = np.zeros_like(all_predictions[0])
        for pred, weight in zip(all_predictions, weights):
            ensemble_pred += pred * weight

        final_predictions = np.argmax(ensemble_pred, axis=1)

        return final_predictions

    def train_and_predict(self, train_path, test_path):
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
        y_train = train_fe['label'] if 'label' in train_fe.columns else None
        groups = train_fe['WELL'] if 'WELL' in train_fe.columns else None
        X_test = test_fe[self.feature_columns]

        print(f"训练数据特征形状: {X_train.shape}")

        # 最终数据清理
        print("进行最终数据清理...")
        X_train = self.robust_clean_data(X_train)
        X_test = self.robust_clean_data(X_test)

        # 检查y_train是否存在
        if y_train is None:
            print("错误: 训练数据中没有找到label列")
            return None
        
        # 【可选】样本平衡（取消注释以启用）
        # X_train, y_train = self.balance_samples(X_train, y_train)

        # 划分验证集
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

        # 训练模型
        tree_models = self.train_tree_models_advanced(X_train, y_train, groups)
        neural_models = self.train_neural_models(X_train_nn, y_train_nn, X_val_nn, y_val_nn)

        # 集成预测
        test_predictions = self.ensemble_predictions_advanced(tree_models, neural_models, X_test)

        # 创建提交文件
        id_column = None
        for col in ['ID', 'id']:
            if col in test_df.columns:
                id_column = col
                break

        if id_column:
            submission_ids = test_df[id_column].values
        else:
            submission_ids = range(len(test_predictions))

        submission = pd.DataFrame({
            'id': submission_ids,
            'predict': test_predictions
        })

        return submission

# 主程序
if __name__ == "__main__":
    identifier = AdvancedLithologyIdentifier()

    try:
        submission = identifier.train_and_predict(
            "train.csv",
            "validation_without_label.csv"
        )

        if submission is not None:
            submission.to_csv("submission_advanced.csv", index=False)
            print("\n预测完成！结果已保存到 submission_advanced.csv")
            print(f"提交文件形状: {submission.shape}")
            print(f"预测值分布:\n{submission['predict'].value_counts().sort_index()}")
        else:
            print("训练失败，无法生成预测结果")

    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()
