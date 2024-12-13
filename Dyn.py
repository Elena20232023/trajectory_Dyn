import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm

# ==================== 缓存数据与模型 ====================#
@st.cache_data
def load_data(file_path):
    return pd.read_excel(file_path)

@st.cache_resource
def fit_model(X, y):
    X_sm = sm.add_constant(X)
    return sm.Logit(y, X_sm).fit(disp=False)

data_path = 'Dyn.xlsx'
data = load_data(data_path)

target_var = 'trajectory'
feature_names = [
    'Disease counts',
    'Age',
    'ADL self-rating',
    'BMI',
    'Hearing ability',
    'Life expectancy',
    'Health status'
]

X = data[feature_names]
y = data[target_var]
model = fit_model(X, y)

params = model.params
intercept = params['const']
coefs = params.drop('const').values

# 定义取值范围与映射
var_ranges = {
    'Disease counts': range(0, 16),
    'Age': [1, 2, 3],
    'ADL self-rating': [1, 2, 3],
    'BMI': [1, 2, 3, 4],
    'Hearing ability': [1, 2, 3, 4, 5],
    'Life expectancy': [1, 2, 3, 4, 5],
    'Health status': [1, 2, 3, 4, 5]
}

var_value_labels = {
    'Disease counts': None,
    'Age': {1: '45-59 year', 2: '60-79 year', 3: '≥80 year'},
    'ADL self-rating': {1: '20', 2: '21-40', 3: '>40'},
    'BMI': {1: '<18.5 kg/m²', 2: '18.5-23.9 kg/m²', 3: '24-27.9 kg/m²', 4: '≥28 kg/m²'},
    'Hearing ability': {1: 'Excellent', 2: 'Very good', 3: 'Good', 4: 'Fair', 5: 'Poor'},
    'Life expectancy': {1: 'Almost impossible', 2: 'Not very likely', 3: 'Maybe', 4: 'Very likely', 5: 'Almost certain'},
    'Health status': {1: 'Very good', 2: 'Good', 3: 'Fair', 4: 'Poor', 5: 'Very poor'}
}

def compute_lp(values, intercept, coefs):
    return intercept + np.sum(coefs * values)

baseline = np.array([min(var_ranges[f]) for f in feature_names])
baseline_lp = compute_lp(baseline, intercept, coefs)

# ==================== 绘图函数 ====================#
def plot_nomogram(selected_values):
    # 将编码转换为具体的文本标签
    display_values = {}
    for var, value in selected_values.items():
        if var_value_labels[var]:  # 如果有映射
            display_values[var] = var_value_labels[var][value]
        else:  # 如果没有映射
            display_values[var] = value

    fig, ax = plt.subplots(figsize=(8, 4))  # 减小图形尺寸
    plt.rcParams['font.size'] = 11

    # 计算线性预测值（LP）和概率
    chosen_values_array = baseline.copy()
    for i, f in enumerate(feature_names):
        chosen_values_array[i] = selected_values[f]
    chosen_lp = compute_lp(chosen_values_array, intercept, coefs)
    pred_prob = 1 / (1 + np.exp(-chosen_lp))

    # 计算95%置信区间
    standard_errors = model.bse
    z_score = norm.ppf(0.975)  # 95%置信水平
    lower_bound = pred_prob - z_score * standard_errors.mean()
    upper_bound = pred_prob + z_score * standard_errors.mean()

    # 绘制图表
    ax.errorbar(
        pred_prob,
        0.5,
        xerr=[[pred_prob - lower_bound], [upper_bound - pred_prob]],
        fmt='o',
        color='blue',
        capsize=5,
        label='Prediction'
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Probability')
    ax.set_title('Graphical Summary')

    # 添加网格线
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)


    # 在图形右侧显示变量和预测结果
    text_x = 1.1
    text_y = 0.9
    ax.text(
        text_x,
        text_y,
        f"Selected Variables:\n" +
        "\n".join([f"{k}: {v}" for k, v in display_values.items()]),
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top'
    )
    ax.text(
        text_x,
        text_y - 0.6,
        f"Prediction: {pred_prob:.4f}\n95% CI: [{lower_bound:.4f}, {upper_bound:.4f}]",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        color='blue'
    )

    ax.legend()
    plt.tight_layout()
    return fig, pred_prob, lower_bound, upper_bound

# ==================== 页面布局 ====================#
st.title("Dynamic Nomogram")

st.sidebar.header("Input Parameters")

selected_values = {}

# Disease counts 用 slider
selected_values['Disease counts'] = st.sidebar.slider(
    "Disease counts",
    min_value=0,
    max_value=14,
    value=0
)

# 其他分类变量用下拉菜单
for f in ['Age', 'ADL self-rating', 'BMI', 'Hearing ability', 'Life expectancy', 'Health status']:
    mapping = var_value_labels[f]
    options = [mapping[v] for v in var_ranges[f]]
    default_idx = 0
    chosen_label = st.sidebar.selectbox(f"{f}:", options, index=default_idx)
    reverse_mapping = {mapping[v]: v for v in var_ranges[f]}
    selected_val = reverse_mapping[chosen_label]
    selected_values[f] = selected_val

predict_button = st.sidebar.button("Predict")

# 确保只有在用户点击 Predict 按钮后才运行预测逻辑
if predict_button:
    fig, predicted_probability, ci_lower, ci_upper = plot_nomogram(selected_values)

    tabs = st.tabs(["Graphical Summary", "Numerical Summary", "Model Summary"])

    with tabs[0]:
        st.subheader("95% Confidence Interval for Response")
        st.pyplot(fig)

    with tabs[1]:
        st.write(f"**Predicted Probability:** {predicted_probability*100:.4f}%")
        st.write(f"**95% CI:** [{ci_lower*100:.4f}%, {ci_upper*100:.4f}%]")

    with tabs[2]:
        st.write("**Model Summary:**")
        st.text(model.summary())
else:
    st.write("Click **Predict** to generate the dynamic nomogram.")


st.sidebar.write("Press Quit to exit the application")
quit_button = st.sidebar.button("Quit")
if quit_button:
    st.stop()


# cd E:\1-Python\0_Heterogeneity_trajectory\CODE\2_new_code
# Start-Process -WindowStyle Hidden -FilePath "streamlit" -ArgumentList "run", "app.py"
