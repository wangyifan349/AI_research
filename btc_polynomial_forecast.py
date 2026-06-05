import os  # 用于创建输出目录、拼接文件路径
import requests  # 用于请求公开 API
import numpy as np  # 用于数值计算
import pandas as pd  # 用于处理表格数据
import matplotlib.pyplot as plt  # 用于绘图和保存图片

from sklearn.pipeline import make_pipeline  # 用于组合多项式特征和线性回归模型
from sklearn.preprocessing import PolynomialFeatures  # 用于生成多项式特征
from sklearn.linear_model import LinearRegression  # 线性回归模型
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error  # 模型评估指标


OUTPUT_DIR = "btc_forecast_output"  # 保存 CSV 和图片的输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 如果目录不存在则创建，存在则不报错


def fetch_btc_prices(days=365, vs_currency="usd"):  # 获取 BTC 历史价格数据
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"  # CoinGecko 历史价格 API

    params = {  # API 请求参数
        "vs_currency": vs_currency,  # 计价货币，例如 usd
        "days": days,  # 获取最近多少天的数据
        "interval": "daily"  # 使用日线数据
    }

    response = requests.get(url, params=params, timeout=30)  # 发送 GET 请求
    response.raise_for_status()  # 如果请求失败，抛出异常

    data = response.json()  # 将返回结果解析为 JSON

    df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])  # 提取价格数据并转成 DataFrame
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date  # 将毫秒时间戳转成日期
    df["date"] = pd.to_datetime(df["date"])  # 将 date 字段转成 pandas datetime 类型

    df = (  # 清洗并排序数据
        df[["date", "price"]]  # 只保留日期和价格两列
        .drop_duplicates("date")  # 删除重复日期
        .sort_values("date")  # 按日期升序排列
        .reset_index(drop=True)  # 重置索引
    )

    return df  # 返回清洗后的历史价格数据


def evaluate_model(y_true, y_pred):  # 计算模型拟合效果
    mae = mean_absolute_error(y_true, y_pred)  # 平均绝对误差
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # 均方根误差
    r2 = r2_score(y_true, y_pred)  # R² 拟合优度

    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # 平均绝对百分比误差

    return {  # 返回评估指标
        "r2": r2,  # R²
        "mae": mae,  # MAE
        "rmse": rmse,  # RMSE
        "mape": mape  # MAPE
    }


def train_predict(df, train_days, forecast_days, degree=3):  # 训练模型并预测未来价格
    train_df = df.tail(train_days).copy().reset_index(drop=True)  # 取最近 train_days 天作为训练数据

    X_train = np.arange(len(train_df)).reshape(-1, 1)  # 用第几天作为特征 X
    y_train = train_df["price"].values  # 价格作为目标变量 y

    model = make_pipeline(  # 创建多项式回归管道
        PolynomialFeatures(degree=degree),  # 将时间特征扩展为 degree 阶多项式
        LinearRegression()  # 对多项式特征做线性回归
    )

    model.fit(X_train, y_train)  # 训练模型

    fitted_price = model.predict(X_train)  # 对训练区间进行拟合预测

    metrics = evaluate_model(y_train, fitted_price)  # 计算拟合效果指标

    X_future = np.arange(  # 构造未来预测区间的时间特征
        len(train_df),  # 从训练数据结束后的下一天开始
        len(train_df) + forecast_days  # 一直到未来 forecast_days 天
    ).reshape(-1, 1)  # 转成 sklearn 需要的二维数组

    predicted_price = model.predict(X_future)  # 预测未来价格

    future_dates = pd.date_range(  # 构造未来日期序列
        start=train_df["date"].iloc[-1] + pd.Timedelta(days=1),  # 从最后一个训练日期的下一天开始
        periods=forecast_days,  # 生成 forecast_days 个日期
        freq="D"  # 每天一个点
    )

    fitted_df = pd.DataFrame({  # 保存训练区间的真实价格和拟合价格
        "date": train_df["date"],  # 日期
        "actual_price": train_df["price"],  # 真实价格
        "fitted_price": fitted_price  # 拟合价格
    })

    forecast_df = pd.DataFrame({  # 保存未来预测结果
        "date": future_dates,  # 未来日期
        "predicted_price": predicted_price  # 预测价格
    })

    return fitted_df, forecast_df, metrics  # 返回拟合数据、预测数据和评估指标


def print_report(name, metrics, forecast_df):  # 打印模型评估和预测结果
    print("\n" + "=" * 70)  # 打印分隔线
    print(name)  # 打印实验名称
    print("=" * 70)  # 打印分隔线

    print(f"R²   : {metrics['r2']:.6f}")  # 打印 R²
    print(f"MAE  : ${metrics['mae']:,.2f}")  # 打印 MAE
    print(f"RMSE : ${metrics['rmse']:,.2f}")  # 打印 RMSE
    print(f"MAPE : {metrics['mape']:.2f}%")  # 打印 MAPE

    print("\n预测结果：")  # 打印预测结果标题
    print(f"预测起始日期 : {forecast_df['date'].iloc[0].date()}")  # 打印预测开始日期
    print(f"预测结束日期 : {forecast_df['date'].iloc[-1].date()}")  # 打印预测结束日期
    print(f"最后一天预测价格 : ${forecast_df['predicted_price'].iloc[-1]:,.2f}")  # 打印最后一天预测价格
    print(f"预测最高价 : ${forecast_df['predicted_price'].max():,.2f}")  # 打印预测区间最高价
    print(f"预测最低价 : ${forecast_df['predicted_price'].min():,.2f}")  # 打印预测区间最低价

    print("\n前 10 天预测：")  # 打印前 10 天预测结果
    print(forecast_df.head(10).to_string(index=False))  # 以表格形式打印前 10 行

    print("\n后 10 天预测：")  # 打印后 10 天预测结果
    print(forecast_df.tail(10).to_string(index=False))  # 以表格形式打印后 10 行


def plot_and_save(title, fitted_df, forecast_df, filename):  # 绘图并保存图片
    plt.figure(figsize=(14, 7))  # 创建画布并设置尺寸

    plt.plot(  # 绘制真实价格曲线
        fitted_df["date"],  # x 轴为日期
        fitted_df["actual_price"],  # y 轴为真实价格
        label="Actual price"  # 图例名称
    )

    plt.plot(  # 绘制拟合价格曲线
        fitted_df["date"],  # x 轴为日期
        fitted_df["fitted_price"],  # y 轴为拟合价格
        label="Fitted price"  # 图例名称
    )

    plt.plot(  # 绘制未来预测价格曲线
        forecast_df["date"],  # x 轴为未来日期
        forecast_df["predicted_price"],  # y 轴为预测价格
        label="Forecast price"  # 图例名称
    )

    plt.title(title)  # 设置图表标题
    plt.xlabel("Date")  # 设置 x 轴名称
    plt.ylabel("BTC Price USD")  # 设置 y 轴名称
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格
    plt.tight_layout()  # 自动调整布局

    path = os.path.join(OUTPUT_DIR, filename)  # 拼接图片保存路径
    plt.savefig(path, dpi=200)  # 保存图片
    plt.show()  # 显示图片

    print(f"图表已保存: {path}")  # 打印图片保存路径


def save_csv(fitted_df, forecast_df, prefix):  # 保存拟合数据和预测数据
    fitted_path = os.path.join(OUTPUT_DIR, f"{prefix}_fitted.csv")  # 拟合数据 CSV 路径
    forecast_path = os.path.join(OUTPUT_DIR, f"{prefix}_forecast.csv")  # 预测数据 CSV 路径

    fitted_df.to_csv(fitted_path, index=False)  # 保存拟合数据 CSV
    forecast_df.to_csv(forecast_path, index=False)  # 保存预测数据 CSV

    print(f"拟合数据已保存: {fitted_path}")  # 打印拟合数据保存路径
    print(f"预测数据已保存: {forecast_path}")  # 打印预测数据保存路径


if __name__ == "__main__":  # 程序入口
    degree = 3  # 多项式阶数

    df = fetch_btc_prices(days=365)  # 获取最近一年 BTC 日线价格

    print("原始数据概览：")  # 打印原始数据标题
    print(df.tail())  # 打印最近几行数据
    print("\n价格统计：")  # 打印价格统计标题
    print(df["price"].describe())  # 打印价格描述性统计

    experiments = [  # 定义多组训练和预测实验
        {
            "name": "最近半年训练，预测未来半年",  # 实验名称
            "train_days": 180,  # 使用最近 180 天训练
            "forecast_days": 180,  # 预测未来 180 天
            "prefix": "train_180_forecast_180",  # 输出文件名前缀
            "image": "train_180_forecast_180.png"  # 输出图片文件名
        },
        {
            "name": "最近半年训练，预测未来一年",  # 实验名称
            "train_days": 180,  # 使用最近 180 天训练
            "forecast_days": 365,  # 预测未来 365 天
            "prefix": "train_180_forecast_365",  # 输出文件名前缀
            "image": "train_180_forecast_365.png"  # 输出图片文件名
        },
        {
            "name": "最近一年训练，预测未来半年",  # 实验名称
            "train_days": 365,  # 使用最近 365 天训练
            "forecast_days": 180,  # 预测未来 180 天
            "prefix": "train_365_forecast_180",  # 输出文件名前缀
            "image": "train_365_forecast_180.png"  # 输出图片文件名
        },
        {
            "name": "最近一年训练，预测未来一年",  # 实验名称
            "train_days": 365,  # 使用最近 365 天训练
            "forecast_days": 365,  # 预测未来 365 天
            "prefix": "train_365_forecast_365",  # 输出文件名前缀
            "image": "train_365_forecast_365.png"  # 输出图片文件名
        }
    ]

    for exp in experiments:  # 逐个执行实验
        fitted_df, forecast_df, metrics = train_predict(  # 训练模型并得到预测结果
            df=df,  # 输入历史价格数据
            train_days=exp["train_days"],  # 当前实验的训练天数
            forecast_days=exp["forecast_days"],  # 当前实验的预测天数
            degree=degree  # 多项式阶数
        )

        print_report(exp["name"], metrics, forecast_df)  # 打印当前实验报告

        save_csv(  # 保存当前实验 CSV
            fitted_df=fitted_df,  # 拟合数据
            forecast_df=forecast_df,  # 预测数据
            prefix=exp["prefix"]  # 文件名前缀
        )

        plot_and_save(  # 绘制并保存当前实验图表
            title=f"{exp['name']} | Polynomial Degree {degree}",  # 图表标题
            fitted_df=fitted_df,  # 拟合数据
            forecast_df=forecast_df,  # 预测数据
            filename=exp["image"]  # 图片文件名
        )
