import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from recommender_core import MovieLensRecommender

# === 新增这两行，解决图表中文显示方块的问题 ===
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签（Windows系统可用黑体）
# plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
# ============================================

# ================= 页面配置 =================
st.set_page_config(page_title="工业级推荐系统可视化演示", layout="wide")
st.title("🎬 电影推荐系统")


# ================= 核心模型加载 (使用缓存避免重复训练) =================
@st.cache_resource(show_spinner="正在拉取数据并训练模型，请稍候 (首次运行约需15秒)...")
def load_and_train_model():
    recsys = MovieLensRecommender()
    recsys.load_and_preprocess_data()
    recsys.train_models()
    return recsys


recsys = load_and_train_model()

# ================= 侧边栏：用户选择 =================
st.sidebar.header("🕹️ 控制面板")
# 获取所有有效用户
valid_users = recsys.user_item_matrix.index.tolist()
selected_user = st.sidebar.selectbox("请选择一个体验用户 (User ID)", options=valid_users[:50])  # 取前50个演示

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ 算法漏斗参数")
top_n_recall = st.sidebar.slider("单路召回数量 (Recall)", min_value=10, max_value=100, value=50, step=10)
top_k_recommend = st.sidebar.slider("最终推荐数量 (Top-K)", min_value=3, max_value=15, value=5, step=1)

# ================= 主体内容区 =================
if selected_user:
    st.markdown(f"### 👤 用户 {selected_user} 的历史偏好 (Top 5 评分)")

    # 1. 提取用户历史高分电影
    user_history = recsys.ratings[recsys.ratings['userId'] == selected_user]
    top_history = user_history.sort_values(by='rating', ascending=False).head(5)

    history_details = top_history.merge(recsys.movies, on='movieId')[['title', 'rating']]
    st.dataframe(history_details, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔍 漏斗阶段剖析")

    col1, col2, col3 = st.columns(3)

    # ================= 阶段 1: 召回 =================
    recalled_items = recsys.recall(selected_user, top_n=top_n_recall)
    col1.info(f"**阶段一：多路召回**\n\nCF与MF合并去重后，从海量库中粗筛出 **{len(recalled_items)}** 部候选影片。")

    # ================= 阶段 2: 精排 =================
    ranked_list = recsys.rank(selected_user, recalled_items)
    col2.warning(f"**阶段二：综合精排**\n\n结合预测得分与热度对 {len(recalled_items)} 部影片进行综合打分并排序。")

    # ================= 阶段 3: 重排 =================
    final_recs_ids = []
    watched_movies = set(user_history['movieId'])

    for item_id, score in ranked_list:
        if item_id not in watched_movies:
            final_recs_ids.append((item_id, score))
        if len(final_recs_ids) >= top_k_recommend:
            break

    col3.success(f"**阶段三：业务重排**\n\n剔除用户已看过的影片，最终输出 Top **{len(final_recs_ids)}** 给用户。")

    st.markdown("---")
    st.markdown(f"### 🎉 最终为您推荐的 {top_k_recommend} 部电影")

    # 提取最终推荐的详细信息用于绘图
    plot_data = []
    for mid, score in final_recs_ids:
        title = recsys.movies[recsys.movies['movieId'] == mid]['title'].values
        title = title[0] if len(title) > 0 else f"Unknown ({mid})"
        plot_data.append({'Movie': title, 'Score': score})

    df_plot = pd.DataFrame(plot_data)

    # 左右两栏布局：左侧显示列表，右侧显示得分柱状图
    res_col1, res_col2 = st.columns([1, 1.5])

    with res_col1:
        # 显示精美列表
        for i, row in df_plot.iterrows():
            st.markdown(f"**{i + 1}. {row['Movie']}** (综合得分: `{row['Score']:.2f}`)")
        
    # with res_col2:
    #     st.markdown("##### 📊 预测得分对比")
    #     # 将 DataFrame 的索引设为电影名，这样图表的 X 轴就是电影名
    #     chart_data = df_plot.set_index('Movie')
    #     # 使用 Streamlit 自带的交互式柱状图，彻底解决云端中文乱码问题
    #     st.bar_chart(chart_data)
    with res_col2:
        # 使用 Plotly 绘制高颜值、带渐变色的交互式水平柱状图
        fig = px.bar(
            df_plot, 
            x='Score', 
            y='Movie', 
            orientation='h', # 水平显示
            color='Score',   # 根据得分映射颜色
            color_continuous_scale='viridis', # 恢复你喜欢的旧版高级渐变色
            text='Score'     # 在柱子上显示具体数值
        )
        
        # 优化图表排版细节
        fig.update_layout(
            title="<b>📊 预测得分对比</b>",
            xaxis_title="预测综合得分",
            yaxis_title="",
            yaxis={'categoryorder':'total ascending'}, # 让高分排在最上面
            plot_bgcolor='rgba(0,0,0,0)', # 背景透明，更贴合网页主题
            height=400
        )
        
        # 调整柱子上的文字显示格式
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        
        # 将 Plotly 图表渲染到 Streamlit 页面上
        st.plotly_chart(fig, use_container_width=True)
