import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression

def load_data(file_path):
    """加载原始数据并初步清洗"""
    df = pd.read_csv(file_path)
    # 删除缺失值
    df.dropna(inplace=True)
    # 筛选关键列（示例）
    df = df[["material", "adsorption_energy", "porosity", "capacity_retention"]]
    return df

def preprocess_data(df):
    """数据预处理：标准化与特征选择"""
    # 标准化吸附能与孔隙率
    scaler = StandardScaler()
    df[["adsorption_energy", "porosity"]] = scaler.fit_transform(df[["adsorption_energy", "porosity"]])
    
    # 特征选择（基于目标变量容量保持率）
    X = df[["adsorption_energy", "porosity"]]
    y = df["capacity_retention"]
    selector = SelectKBest(score_func=f_regression, k=2)
    X_selected = selector.fit_transform(X, y)
    
    # 合并处理后的数据
    df_processed = pd.DataFrame(X_selected, columns=["adsorption_energy_norm", "porosity_norm"])
    df_processed["capacity_retention"] = y.values
    return df_processed

def split_data(df, test_size=0.2, val_size=0.1):
    """划分训练集、验证集、测试集（7:2:1）"""
    train_val, test = train_test_split(df, test_size=test_size, random_state=42)
    train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=42)
    return train, val, test

if __name__ == "__main__":
    raw_df = load_data("data/raw_data.csv")
    processed_df = preprocess_data(raw_df)
    train, val, test = split_data(processed_df)
    processed_df.to_csv("data/processed_data.csv", index=False)
