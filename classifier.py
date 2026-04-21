'''
Класифікація новин на "Достовірні" / "Фейк" методом навчання із вчителем.
TF-IDF векторизація + SVM класифікатор.
'''

import os
import warnings
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.set_loglevel('error')
warnings.filterwarnings('ignore')
logging.getLogger('matplotlib').setLevel(logging.ERROR)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

# Векторизатор та класифікатор
vectorizer = TfidfVectorizer(max_df=0.7, min_df=1, max_features=5000)
clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')


def train_classifier(texts_processed, labels):
    '''
    Навчає TF-IDF + Logistic Regression класифікатор на розміченому наборі даних.
    '''
    global vectorizer, clf

    print('\nКрок 1: TF-IDF векторизація:')
    X = vectorizer.fit_transform(texts_processed)
    y = np.array(labels)
    print(f'  Розмір матриці ознак: {X.shape}')

    print('Крок 2: Поділ на навчальну/тестову вибірки (70/30):')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f'  Навчальна: {X_train.shape[0]} зразків')
    print(f'  Тестова:   {X_test.shape[0]} зразків')

    print('Крок 3: Навчання SVM (kernel=linear).')
    clf.fit(X_train, y_train)

    print('Крок 4: Оцінка якості.')
    y_pred = clf.predict(X_test)
    return X_test, y_test, y_pred


def evaluate_classifier(y_test, y_pred, results_dir):
    '''
    Виводить метрики якості та зберігає текстовий звіт у results/.
    '''
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(
        y_test, y_pred,
        target_names=['Достовірна', 'Фейк']
    )

    # Вивід у консоль
    print('\n' + '='*50)
    print(f'  Accuracy (точність): {accuracy:.4f} ({accuracy*100:.1f}%)')
    print('='*50)
    print('Classification Report:')
    print(report)

    msg    = ('перевищує' if accuracy >= 0.60 else 'нижча за')
    print(f'  Точність {accuracy*100:.1f}% {msg} мінімальний поріг 60%')

    # Збереження текстового звіту у файл
    report_path = os.path.join(results_dir, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('ЗВІТ КЛАСИФІКАТОРА SVM\n')
        f.write('='*50 + '\n')
        f.write(f'Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)\n')
        f.write('='*50 + '\n\n')
        f.write('Classification Report:\n')
        f.write(report)

    return accuracy


def predict_news(text_processed):
    '''
    Класифікує один текст новини. Повертає: "Достовірна" або "Фейк".
    '''
    vec        = vectorizer.transform([text_processed])
    prediction = clf.predict(vec)[0]
    return 'Фейк' if prediction == 1 else 'Достовірна'


def plot_tfidf_top_terms(results_dir, n=15):
    '''
    Будує та зберігає графік топ-N TF-IDF термінів у results/.
    '''
    feature_names = vectorizer.get_feature_names_out()
    idf_scores    = vectorizer.idf_
    top_indices   = np.argsort(idf_scores)[::-1][:n]
    top_terms     = [feature_names[i] for i in top_indices]
    top_scores    = [idf_scores[i]    for i in top_indices]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.barh(top_terms[::-1], top_scores[::-1], color='steelblue')
    ax.set_xlabel('IDF вага')
    ax.set_title(f'Топ-{n} термінів за TF-IDF вагою')
    plt.tight_layout()
    path = os.path.join(results_dir, 'tfidf_top_terms.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)