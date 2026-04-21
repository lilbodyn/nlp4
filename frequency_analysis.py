'''
Частотний та ймовірнісний аналіз тексту (Рівень ІІ).
- TF-IDF аналіз
- Лексична дисперсія
- Розподіл довжини слів
- Біграмний аналіз (граф спів-входжень)
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

import networkx as nx
from collections import Counter
from nltk import FreqDist, ConditionalFreqDist
from nltk.draw.dispersion import dispersion_plot


def tfidf_frequency_analysis(token_list, results_dir, top_n=20):
    '''
    Частотний аналіз токенів: термінальна частота (TF).
    Зберігає два графіки у results/.
    '''
    print(f'\nTF - розподіл термінів (топ {top_n})')
    freq_dist = FreqDist(token_list)
    print(f'  Унікальних термінів: {len(freq_dist)}')
    print(f'  Топ-20: {freq_dist.most_common(20)}')

    # Графік термінальної частоти
    freq_dist.plot(top_n, cumulative=False,
                   title=f'Термінальна частота TF - топ {top_n}')
    plt.tight_layout()
    path = os.path.join(results_dir, 'tf_frequency.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close('all')

    # Графік кумулятивної частоти
    freq_dist.plot(top_n, cumulative=True,
                   title=f'Кумулятивна частота TF - топ {top_n}')
    plt.tight_layout()
    path = os.path.join(results_dir, 'tf_cumulative.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close('all')

    return freq_dist


def lexical_dispersion(token_list, results_dir, top_n=10):
    '''
    Лексична дисперсія - позиційний розподіл топ-N слів у тексті.
    Зберігає графік у results/.
    '''
    print(f'\nЛексична дисперсія (топ {top_n} слів)')
    if len(token_list) < top_n:
        print('  Недостатньо токенів для аналізу дисперсії')
        return

    freq_dist = FreqDist(token_list)
    targets   = [word for word, _ in freq_dist.most_common(top_n)]
    print(f'  Цільові слова: {targets}')

    try:
        dispersion_plot(token_list, targets,
                        title='Лексична дисперсія - топ слів',
                        ignore_case=True)
        plt.tight_layout()
        path = os.path.join(results_dir, 'lexical_dispersion.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close('all')
    except Exception as e:
        print(f'  Помилка побудови дисперсії: {e}')


def word_length_distribution(token_list, results_dir):
    '''
    Розподіл довжини слів - гістограма та лінійний графік.
    Зберігає два графіки у results/.
    '''
    print('\nРозподіл довжини слів')
    lengths = [len(token) for token in token_list if token.isalpha()]
    if not lengths:
        print('  Немає даних для розподілу довжини')
        return

    length_counter = Counter(lengths)
    print(f'  Мін. довжина: {min(lengths)} символів')
    print(f'  Макс. довжина: {max(lengths)} символів')
    print(f'  Середня довжина: {sum(lengths)/len(lengths):.2f} символів')

    # Гістограма
    sorted_lengths = sorted(length_counter.keys())
    counts = [length_counter[l] for l in sorted_lengths]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(sorted_lengths, counts, color='teal', alpha=0.8, edgecolor='white')
    ax.set_xlabel('Довжина слова (символів)')
    ax.set_ylabel('Кількість слів')
    ax.set_title('Розподіл довжини слів у корпусі')
    ax.set_xticks(sorted_lengths)
    plt.tight_layout()
    path = os.path.join(results_dir, 'word_length_hist.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ConditionalFreqDist лінійний графік
    cfd = ConditionalFreqDist(
        ('всі слова', len(token))
        for token in token_list
        if token.isalpha()
    )
    cfd.plot(title='Розподіл довжини слів (ConditionalFreqDist)')
    plt.tight_layout()
    path = os.path.join(results_dir, 'word_length_cfd.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close('all')


def bigram_analysis(token_list, results_dir, top_n=30):
    '''
    Біграмний аналіз - граф спів-входжень слів.
    Зберігає граф у results/.
    '''
    print(f'\nБіграмний аналіз (топ {top_n} слів)')
    if len(token_list) < 5:
        print('  Недостатньо токенів для біграмного аналізу')
        return

    freq_dist  = FreqDist(token_list)
    top_words  = set(word for word, _ in freq_dist.most_common(top_n))
    filtered   = [t for t in token_list if t in top_words]
    bigrams    = list(zip(filtered[:-1], filtered[1:]))
    bigram_ctr = Counter(bigrams)
    print(f'  Топ-30 біграм: {bigram_ctr.most_common(30)}')

    G = nx.Graph()
    for (w1, w2), count in bigram_ctr.most_common(40):
        G.add_edge(w1, w2, weight=count)

    fig, ax = plt.subplots(figsize=(14, 8))
    pos          = nx.spring_layout(G, seed=42, k=1.4)
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_w        = max(edge_weights) if edge_weights else 1

    nx.draw_networkx_nodes(G, pos, node_color='steelblue',
                           node_size=700, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9,
                            font_color='white', font_weight='bold', ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.5,
                           width=[2 * w / max_w for w in edge_weights],
                           edge_color='gray', ax=ax)
    ax.set_title('Граф біграмних спів-входжень слів')
    ax.axis('off')
    plt.margins(0.15)
    plt.tight_layout()
    path = os.path.join(results_dir, 'bigram_graph.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def run_frequency_analysis(token_list, results_dir):
    '''
    Запускає повний частотний аналіз (Рівень ІІ).
    '''

    tfidf_frequency_analysis(token_list, results_dir)
    lexical_dispersion(token_list, results_dir)
    word_length_distribution(token_list, results_dir)
    bigram_analysis(token_list, results_dir)

    print('\nЧастотний та ймовірнісний аналіз завершено!')