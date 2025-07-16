import pandas as pd
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


# 自定义特征加权变换器 - 必须与训练时使用的类定义完全相同
class DynamicFeatureWeighter(BaseEstimator, TransformerMixin):
    """动态调整特征权重的变换器

    注意: max_token_attention_dq已经被缩放了10倍
    """

    def __init__(self, weight_max_token=0.5):
        self.weight_max_token = weight_max_token
        # 确保权重和为1
        self.weight_entropy = 1.0 - self.weight_max_token

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # 创建副本避免修改原始数据
        X_transformed = X.copy()

        # 应用权重
        X_transformed['max_token_attention_dq'] = X_transformed['max_token_attention_dq'] * self.weight_max_token
        X_transformed['entropy_dq'] = X_transformed['entropy_dq'] * self.weight_entropy

        return X_transformed


def create_model(weight_max_token=0.7):
    """创建与训练时相同的模型管道"""
    model = Pipeline([
        ('weighter', DynamicFeatureWeighter(weight_max_token=weight_max_token)),
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            eval_metric='logloss'
        ))
    ])
    return model


def train_model(model, train_file='document_attention_features.csv'):
    """使用训练数据训练模型"""
    if not os.path.exists(train_file):
        print(f"警告: 训练数据文件 {train_file} 不存在，无法训练模型")
        return False

    print(f"加载训练数据: {train_file}")
    train_df = pd.read_csv(train_file)

    # 准备训练数据
    train_df['label'] = train_df['doc_type'].map({'poison': 1, 'clean': 0})
    X_train = train_df[['max_token_attention_dq', 'entropy_dq']].copy()
    y_train = train_df['label']

    # 将max_token_attention_dq的数值乘以10
    X_train['max_token_attention_dq'] = X_train['max_token_attention_dq'] * 10

    # 训练模型
    print("训练模型...")
    model.fit(X_train, y_train)
    print("模型训练完成")
    return True

def predict_single_sample(model, max_token_attention, entropy):
    """预测单个样本的标签"""
    # 创建特征DataFrame
    features = pd.DataFrame({
        'max_token_attention_dq': [max_token_attention * 10],  # 乘以10，与训练时保持一致
        'entropy_dq': [entropy]
    })

    # 预测
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    # 获取毒性类的概率（第二个元素是毒性类的概率）
    poison_probability = probabilities[1]

    # 转换为文本标签
    label = 'poison' if prediction == 1 else 'clean'

    return label, poison_probability


def main():
    print("\n=== 毒性检测器 ===\n")

    # 创建模型
    print("创建模型...")
    model = create_model()

    # 训练模型
    train_success = train_model(model)
    if not train_success:
        print("警告: 模型未训练，将使用默认参数，结果可能不准确")

    # 交互式预测循环
    print("\n输入特征值进行预测 (输入'q'退出):")

    while True:
        try:
            # 获取用户输入
            max_token_input = input("\nmax_token_attention (例如: 0.85): ")
            if max_token_input.lower() == 'q':
                break

            entropy_input = input("entropy (例如: 6.5): ")
            if entropy_input.lower() == 'q':
                break

            # 转换为浮点数
            max_token_attention = float(max_token_input)
            entropy = float(entropy_input)

            # 预测
            label, probability = predict_single_sample(model, max_token_attention, entropy)

            # 显示结果
            print(f"\n预测结果: {label}")
            # 将概率转换为百分数
            percentage = probability * 100
            print(f"置信度: {percentage:.2f}%")

            # 显示解释
            if label == 'poison':
                confidence_level = "高" if percentage > 80 else "中等" if percentage > 60 else "低"
                print(f"解释: 该样本被预测为有毒，置信度{confidence_level}。")
            else:
                # 对于'clean'标签，我们使用(100-percentage)作为置信度
                clean_confidence = 100 - percentage
                confidence_level = "高" if clean_confidence > 80 else "中等" if clean_confidence > 60 else "低"
                print(f"解释: 该样本被预测为正常，置信度{confidence_level} ({clean_confidence:.2f}%)。")

        except ValueError:
            print("错误: 请输入有效的数字")
        except Exception as e:
            print(f"错误: {e}")

    print("\n感谢使用毒性检测器!")


if __name__ == "__main__":
    main()
