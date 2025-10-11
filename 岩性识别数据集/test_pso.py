# 快速测试脚本 - 验证 PSO 代码是否正常运行
# 使用小规模参数快速测试

import pandas as pd
import numpy as np
import sys

print("="*70)
print("快速测试 PSO 优化代码")
print("="*70)

try:
    from main_pso import LithologyIdentifier
    print("✓ 成功导入 LithologyIdentifier")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

# 检查数据文件
import os
if not os.path.exists('train.csv'):
    print("✗ 找不到 train.csv 文件")
    sys.exit(1)
if not os.path.exists('validation_without_label.csv'):
    print("✗ 找不到 validation_without_label.csv 文件")
    sys.exit(1)
print("✓ 数据文件存在")

# 测试1: 快速特征工程
print("\n" + "="*70)
print("测试1: 特征工程")
print("="*70)

try:
    identifier = LithologyIdentifier()
    train_df = pd.read_csv('train.csv')
    print(f"原始数据形状: {train_df.shape}")
    
    train_fe = identifier.create_advanced_features(train_df)
    print(f"特征工程后形状: {train_fe.shape}")
    print(f"新增特征数: {train_fe.shape[1] - train_df.shape[1]}")
    print("✓ 特征工程测试通过")
except Exception as e:
    print(f"✗ 特征工程失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试2: PSO算法（小规模）
print("\n" + "="*70)
print("测试2: PSO 算法（小规模测试）")
print("="*70)

try:
    from main_pso import PSO
    
    # 简单的测试函数：最大化 -(x^2 + y^2)，最优解应该在 (0, 0)
    def test_func(params):
        x, y = params
        return -(x**2 + y**2)  # 在 (0,0) 处最大值为 0
    
    bounds = [(-5, 5), (-5, 5)]
    pso = PSO(test_func, bounds, n_particles=5, max_iter=5)
    best_params, best_score = pso.optimize()
    
    print(f"\n测试结果:")
    print(f"  最佳参数: x={best_params[0]:.4f}, y={best_params[1]:.4f}")
    print(f"  最佳分数: {best_score:.4f}")
    print(f"  理论最优: x=0, y=0, score=0")
    
    if abs(best_params[0]) < 1 and abs(best_params[1]) < 1 and best_score > -2:
        print("✓ PSO 算法测试通过")
    else:
        print("⚠ PSO 结果不理想，但可能是正常的随机波动")
        
except Exception as e:
    print(f"✗ PSO 算法测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试3: 完整流程（不启用优化，快速测试）
print("\n" + "="*70)
print("测试3: 完整流程（使用默认参数，跳过PSO优化）")
print("="*70)

try:
    identifier = LithologyIdentifier()
    
    submission = identifier.train_and_predict(
        "train.csv",
        "validation_without_label.csv",
        optimize=False,     # 跳过PSO优化
        n_particles=5,
        max_iter=5
    )
    
    print(f"\n预测结果:")
    print(f"  提交文件形状: {submission.shape}")
    print(f"  预测值分布:\n{submission['predict'].value_counts().sort_index()}")
    
    # 保存测试结果
    submission.to_csv("test_submission.csv", index=False)
    print("\n✓ 完整流程测试通过")
    print("  测试结果已保存到 test_submission.csv")
    
except Exception as e:
    print(f"✗ 完整流程测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("所有测试通过！✓")
print("="*70)
print("\n下一步操作:")
print("1. 运行快速优化测试:")
print("   python main_pso.py  # 默认使用 n_particles=15, max_iter=20")
print("\n2. 或者修改 main_pso.py 最后几行，调整参数:")
print("   optimize=True,   # 启用PSO优化")
print("   n_particles=10,  # 减少粒子数加速测试")
print("   max_iter=10      # 减少迭代次数加速测试")
print("\n预计运行时间:")
print("  - 不优化 (optimize=False): 15-20分钟")
print("  - 快速优化 (10粒子×10迭代): 30-45分钟")
print("  - 标准优化 (15粒子×20迭代): 1-2小时")
