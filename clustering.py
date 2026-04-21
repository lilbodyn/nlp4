'''
Внутрішня кластеризація тексту без вчителя (Рівень ІІІ).
K-Means на TF-IDF. Silhouette + Elbow. Візуалізація PCA.
'''

import os
import warnings
import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.set_loglevel('error')

warnings.filterwarnings('ignore')
logging.getLogger('matplotlib').setLevel(logging.ERROR)

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def find_optimal_clusters(X, results_dir, k_range=range(2, 8)):
    '''
    Визначає оптимальне k методами Silhouette та Elbow.
    '''
    print('\nВизначення оптимальної кількості кластерів:')
    silhouette_scores = []
    inertia_values    = []

    for k in k_range:
        km     = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        sil    = silhouette_score(X, labels)
        silhouette_scores.append(sil)
        inertia_values.append(km.inertia_)
        print(f'  k={k}: Silhouette={sil:.4f}, Inertia={km.inertia_:.2f}')

    # Графік Silhouette Score
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(list(k_range), silhouette_scores, 'o-', color='steelblue', linewidth=2)
    ax.set_xlabel('Кількість кластерів (k)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score для визначення оптимального k')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(results_dir, 'silhouette_score.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Графік Elbow Method
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(list(k_range), inertia_values, 'o-', color='coral', linewidth=2)
    ax.set_xlabel('Кількість кластерів (k)')
    ax.set_ylabel('Інерція (Inertia)')
    ax.set_title('Elbow Method - визначення оптимального k')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(results_dir, 'elbow_method.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    optimal_k = list(k_range)[np.argmax(silhouette_scores)]
    print(f'\nОптимальна кількість кластерів (Silhouette): k={optimal_k}')
    return optimal_k


def run_kmeans(X, n_clusters):
    '''
    Запускає K-Means. Повертає мітки та модель.
    '''
    print(f'\nКластеризація: k={n_clusters}...')
    kmeans = KMeans(n_clusters=n_clusters, n_init=10,
                    max_iter=500, random_state=42)
    labels = kmeans.fit_predict(X)
    print('Розподіл по кластерах:')
    for i in range(n_clusters):
        print(f'  Кластер {i}: {np.sum(labels == i)} документів')
    return labels, kmeans


def visualize_clusters(X, labels, texts, n_clusters, results_dir):
    '''
    Візуалізує кластери у 2D (PCA).
    Зберігає у results/.
    '''
    print('\nPCA зниження розмірності для візуалізації:')
    pca  = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X.toarray())
    print(f'  Пояснена дисперсія: {sum(pca.explained_variance_ratio_)*100:.1f}%')

    colors = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple',
              'goldenrod', 'tomato', 'teal', 'slategray']

    fig, ax = plt.subplots(figsize=(12, 7))
    for i in range(n_clusters):
        mask = labels == i
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   s=60, alpha=0.7,
                   color=colors[i % len(colors)],
                   label=f'Кластер {i} ({np.sum(mask)} doc)')
    ax.legend(fontsize=10)
    ax.set_xlabel('PCA Компонента 1')
    ax.set_ylabel('PCA Компонента 2')
    ax.set_title('K-Means кластеризація новин (PCA 2D)')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    path = os.path.join(results_dir, 'kmeans_pca.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_cluster_report(texts, labels, n_clusters, sil_score, results_dir):
    '''
    Зберігає текстовий звіт кластеризації у results/.
    '''
    path = os.path.join(results_dir, 'clustering_report.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('ЗВІТ K-MEANS КЛАСТЕРИЗАЦІЇ\n')
        f.write('='*50 + '\n')
        f.write(f'Кількість кластерів: {n_clusters}\n')
        f.write(f'Silhouette Score:    {sil_score:.4f}\n')
        f.write('='*50 + '\n\n')
        for i in range(n_clusters):
            indices = [j for j, l in enumerate(labels) if l == i]
            f.write(f'--- Кластер {i} ({len(indices)} документів) ---\n')
            for idx in indices[:5]:
                f.write(f'  [{idx}] {texts[idx][:100]}\n')
            f.write('\n')


def run_clustering(texts_processed, texts_original, results_dir):
    '''
    Запускає повну кластеризацію без вчителя (Рівень ІІІ).
    '''

    print('\nTF-IDF векторизація для кластеризації...')
    vec = TfidfVectorizer(max_df=0.8, min_df=1, max_features=3000)
    X   = vec.fit_transform(texts_processed)
    print(f'  Матриця: {X.shape}')

    max_k   = min(8, len(texts_processed) - 1)
    k_range = range(2, max_k + 1)
    optimal_k = find_optimal_clusters(X, results_dir, k_range)

    labels, kmeans = run_kmeans(X, optimal_k)

    final_sil = silhouette_score(X, labels)
    print(f'\nФінальний Silhouette Score: {final_sil:.4f}')

    visualize_clusters(X, labels, texts_original, optimal_k, results_dir)
    save_cluster_report(texts_original, labels, optimal_k, final_sil, results_dir)

    # Виведення прикладів у консоль
    print('\nПриклади текстів по кластерах:')
    for i in range(optimal_k):
        print(f'\n  --- Кластер {i} ---')
        indices = [j for j, l in enumerate(labels) if l == i]
        for idx in indices[:3]:
            print(f'    [{idx}] {texts_original[idx][:80]}...')

    print('\nКластеризацію завершено!')
    return labels, optimal_k, final_sil