# 特征重要性分析工具
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

class FeatureAnalyzer:
    """特征分析工具类"""
    
    def __init__(self, X_train, y_train, feature_names):
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names
        
    def analyze_mutual_information(self, top_k=30):
        """互信息分析"""
        print("\n=== 互信息特征重要性分析 ===")
        
        mi_scores = mutual_info_classif(self.X_train, self.y_train, random_state=42)
        
        feature_scores = pd.DataFrame({
            'feature': self.feature_names,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)
        
        print(f"\nTop {top_k} 最重要特征:")
        print(feature_scores.head(top_k))
        
        # 可视化
        plt.figure(figsize=(12, 8))
        sns.barplot(data=feature_scores.head(top_k), x='mi_score', y='feature')
        plt.title(f'Top {top_k} Features by Mutual Information')
        plt.xlabel('Mutual Information Score')
        plt.tight_layout()
        plt.savefig('feature_importance_mi.png', dpi=300, bbox_inches='tight')
        print("\n已保存: feature_importance_mi.png")
        
        return feature_scores
    
    def analyze_correlation(self, threshold=0.8):
        """分析高度相关的特征（用于特征去冗余）"""
        print("\n=== 特征相关性分析 ===")
        
        corr_matrix = pd.DataFrame(self.X_train, columns=self.feature_names).corr().abs()
        
        # 找出高度相关的特征对
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        if high_corr_pairs:
            high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', ascending=False)
            print(f"\n发现 {len(high_corr_pairs)} 对高度相关特征 (相关系数>{threshold}):")
            print(high_corr_df.head(20))
            
            # 建议删除的特征
            to_remove = set()
            for pair in high_corr_pairs[:20]:  # 只处理前20对
                to_remove.add(pair['feature2'])  # 保留feature1，删除feature2
            
            print(f"\n建议删除的冗余特征 ({len(to_remove)}个):")
            print(list(to_remove))
        else:
            print(f"未发现相关系数>{threshold}的特征对")
        
        return high_corr_pairs
    
    def analyze_variance(self, threshold=0.01):
        """分析低方差特征（接近常数的特征）"""
        print("\n=== 特征方差分析 ===")
        
        variances = np.var(self.X_train, axis=0)
        
        variance_df = pd.DataFrame({
            'feature': self.feature_names,
            'variance': variances
        }).sort_values('variance')
        
        low_variance_features = variance_df[variance_df['variance'] < threshold]
        
        if len(low_variance_features) > 0:
            print(f"\n发现 {len(low_variance_features)} 个低方差特征 (方差<{threshold}):")
            print(low_variance_features)
            print("\n建议删除这些特征（几乎不包含信息）")
        else:
            print(f"未发现方差<{threshold}的特征")
        
        return low_variance_features
    
    def feature_distribution_by_label(self, feature_name):
        """分析特征在不同标签下的分布"""
        df = pd.DataFrame(self.X_train, columns=self.feature_names)
        df['label'] = self.y_train
        
        plt.figure(figsize=(12, 6))
        
        for label in sorted(df['label'].unique()):
            label_data = df[df['label'] == label][feature_name]
            plt.hist(label_data, alpha=0.5, label=f'Class {label}', bins=50)
        
        plt.xlabel(feature_name)
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {feature_name} by Class')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'feature_dist_{feature_name}.png', dpi=300, bbox_inches='tight')
        print(f"\n已保存: feature_dist_{feature_name}.png")
    
    def generate_feature_report(self):
        """生成完整的特征分析报告"""
        print("\n" + "="*60)
        print("特征工程分析报告")
        print("="*60)
        
        print(f"\n特征总数: {len(self.feature_names)}")
        print(f"样本总数: {len(self.X_train)}")
        print(f"类别分布:\n{pd.Series(self.y_train).value_counts().sort_index()}")
        
        # 1. 互信息分析
        mi_scores = self.analyze_mutual_information(top_k=30)
        
        # 2. 相关性分析
        high_corr = self.analyze_correlation(threshold=0.85)
        
        # 3. 方差分析
        low_var = self.analyze_variance(threshold=0.001)
        
        # 4. 统计摘要
        print("\n=== 特征统计摘要 ===")
        df_stats = pd.DataFrame(self.X_train, columns=self.feature_names)
        print(df_stats.describe())
        
        # 保存报告
        report = {
            'mutual_information': mi_scores,
            'high_correlation': pd.DataFrame(high_corr) if high_corr else pd.DataFrame(),
            'low_variance': low_var,
            'statistics': df_stats.describe()
        }
        
        # 保存到Excel
        with pd.ExcelWriter('feature_analysis_report.xlsx') as writer:
            report['mutual_information'].to_excel(writer, sheet_name='Mutual_Information', index=False)
            if not report['high_correlation'].empty:
                report['high_correlation'].to_excel(writer, sheet_name='High_Correlation', index=False)
            report['low_variance'].to_excel(writer, sheet_name='Low_Variance', index=False)
            report['statistics'].to_excel(writer, sheet_name='Statistics')
        
        print("\n已保存完整报告: feature_analysis_report.xlsx")
        
        return report


# 使用示例
if __name__ == "__main__":
    print("特征分析工具")
    print("请在训练主程序中使用此工具")
    print("\n示例代码:")
    print("""
    # 在 main.py 的 train_and_predict 方法中添加：
    
    from feature_analysis import FeatureAnalyzer
    
    # 创建分析器
    analyzer = FeatureAnalyzer(
        X_train.values, 
        y_train.values, 
        X_train.columns.tolist()
    )
    
    # 生成完整报告
    report = analyzer.generate_feature_report()
    
    # 分析特定特征的分布
    analyzer.feature_distribution_by_label('GR')
    analyzer.feature_distribution_by_label('Vsh_larionov')
    """)
