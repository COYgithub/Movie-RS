import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import warnings

warnings.filterwarnings('ignore')


class MovieLensRecommender:
    def __init__(self):
        self.ratings = None
        self.movies = None
        self.user_item_matrix = None

        # 模型参数
        self.user_sim_matrix = None
        self.mf_predictions = None

    # ================= 1. 数据处理 (Data Processing) =================
    def load_and_preprocess_data(self):
        print("=> [1/5] 数据处理：加载 MovieLens-100k 数据集...")

        # 自动下载和解压 ml-100k 数据集
        data_dir = 'ml-100k'
        if not os.path.exists(data_dir):
            print("本地未找到 ml-100k 数据集，正在从 GroupLens 官网下载 (约 5MB)...")
            zip_url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
            urllib.request.urlretrieve(zip_url, 'ml-100k.zip')
            with zipfile.ZipFile('ml-100k.zip', 'r') as zip_ref:
                zip_ref.extractall()
            print("下载并解压完成！")

        # 读取 ratings 数据 (u.data: user id | item id | rating | timestamp. 以 tab 分隔)
        self.ratings = pd.read_csv(
            f'{data_dir}/u.data',
            sep='\t',
            names=['userId', 'movieId', 'rating', 'timestamp']
        )

        # 读取 movies 数据 (u.item: movie id | movie title | ... 以 pipe | 分隔，编码为 latin-1)
        # 我们这里只需要用到前两列：movieId 和 title
        self.movies = pd.read_csv(
            f'{data_dir}/u.item',
            sep='|',
            names=['movieId', 'title'],
            usecols=[0, 1],
            encoding='latin-1'
        )

        # 预处理：MovieLens-100k 数据质量很高，每个用户至少有 20 条评分。
        # 为了进一步降低稀疏度，我们可以做轻量过滤（保留被评价10次以上的电影）
        movie_counts = self.ratings['movieId'].value_counts()
        popular_movies = movie_counts[movie_counts >= 10].index
        self.ratings = self.ratings[self.ratings['movieId'].isin(popular_movies)]

        # 构建用户-物品交互矩阵 (Pivot Table)
        self.user_item_matrix = self.ratings.pivot(
            index='userId', columns='movieId', values='rating'
        ).fillna(0)

        print(f"预处理完成。当前矩阵大小: {self.user_item_matrix.shape[0]} 用户 x {self.user_item_matrix.shape[1]} 电影")

    # ================= 2. 模型训练 (Model Training) =================
    def train_models(self):
        print("=> [2/5] 模型训练：计算协同过滤与矩阵分解...")
        R = self.user_item_matrix.values

        # 模型 A：基于用户的协同过滤 (User-CF)
        self.user_sim_matrix = cosine_similarity(R)

        # 模型 B：矩阵分解 (Matrix Factorization - SVD)
        # 将数据中心化（减去用户平均评分）以消除个人打分习惯偏差
        user_ratings_mean = np.mean(R, axis=1)
        R_demeaned = R - user_ratings_mean.reshape(-1, 1)

        # 提取隐含特征 K=50 (100k数据集稍微丰富一些，可以使用大一点的特征维度)
        U, sigma, Vt = svds(R_demeaned, k=50)
        sigma = np.diag(sigma)

        # 还原预测矩阵：预测评分 = U * Sigma * Vt + user_mean
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
        self.mf_predictions = pd.DataFrame(
            all_user_predicted_ratings,
            columns=self.user_item_matrix.columns,
            index=self.user_item_matrix.index
        )

    # ================= 3. 多路召回 (Multi-channel Recall) =================
    def recall(self, user_id, top_n=50):
        if user_id not in self.user_item_matrix.index:
            return set()

        user_idx = self.user_item_matrix.index.get_loc(user_id)

        # --- 召回路 A: User-CF ---
        # 找最相似的 10 个用户
        similar_users_idx = np.argsort(self.user_sim_matrix[user_idx])[::-1][1:11]
        cf_candidates = set()
        for sim_idx in similar_users_idx:
            # 找到相似用户评分较高 (>=4分) 的电影
            sim_user_ratings = self.user_item_matrix.iloc[sim_idx]
            liked_movies = sim_user_ratings[sim_user_ratings >= 4.0].index.tolist()
            cf_candidates.update(liked_movies)

        # --- 召回路 B: 矩阵分解 (MF) ---
        user_mf_scores = self.mf_predictions.loc[user_id]
        # 取 MF 预测分最高的 Top N
        mf_candidates = set(user_mf_scores.nlargest(top_n).index)

        # 并集
        recalled_items = cf_candidates.union(mf_candidates)
        return recalled_items

    # ================= 4. 排序 (Ranking) =================
    def rank(self, user_id, recalled_items):
        ranked_items = {}
        # 预计算电影热度字典，避免在循环中重复查询 DataFrame
        movie_popularity = self.ratings['movieId'].value_counts().to_dict()

        for item_id in recalled_items:
            mf_score = self.mf_predictions.loc[user_id, item_id]
            popularity = movie_popularity.get(item_id, 0)

            # 综合得分：MF分数占主导，加上热度平滑作为惩罚/奖励因子
            final_score = (0.85 * mf_score) + (0.15 * np.log1p(popularity) / 2)
            ranked_items[item_id] = final_score

        return sorted(ranked_items.items(), key=lambda x: x[1], reverse=True)

    # ================= 5. 重排与过滤 (Re-ranking) =================
    def re_rank(self, user_id, ranked_list, top_k=10):
        watched_movies = set(self.ratings[self.ratings['userId'] == user_id]['movieId'])

        final_recommendations = []
        for item_id, score in ranked_list:
            if item_id in watched_movies:
                continue

            final_recommendations.append((item_id, score))
            if len(final_recommendations) >= top_k:
                break

        return final_recommendations

    def get_movie_title(self, movie_id):
        """辅助方法：通过ID获取电影名"""
        title = self.movies[self.movies['movieId'] == movie_id]['title'].values
        return title[0] if len(title) > 0 else f"Unknown ({movie_id})"


# ================= 终端运行测试 =================
if __name__ == "__main__":
    recsys = MovieLensRecommender()
    recsys.load_and_preprocess_data()
    recsys.train_models()

    # 随机选一个用户进行终端测试
    test_user = recsys.user_item_matrix.index[0]
    print(f"\n================ 为 User {test_user} 生成推荐 ================")

    recalled = recsys.recall(test_user)
    ranked = recsys.rank(test_user, recalled)
    final_res = recsys.re_rank(test_user, ranked, top_k=5)

    print("\n🎉 最终推荐列表:")
    for i, (mid, score) in enumerate(final_res, 1):
        print(f"{i}. {recsys.get_movie_title(mid)} (得分: {score:.3f})")