'''
NLP конвеєр попередньої обробки тексту.
Фільтрація → Нормалізація → Токенізація → Видалення стоп-слів → Лематизація
'''

import re
import spacy

# Завантаження мовної моделі spaCy для української мови
try:
    nlp = spacy.load('uk_core_news_sm')
except OSError:
    nlp = None


def normalize_text(text):
    '''
    Фільтрація та нормалізація тексту.
    Видаляє пунктуацію, цифри, зайві символи.
    Приводить до нижнього регістру.
    '''
    # Видалення HTML-тегів
    text = re.sub(r'<[^>]+>', ' ', text)
    # Видалення цифр
    text = re.sub(r'[0-9]+', ' ', text)
    # Видалення пунктуації та спеціальних символів
    text = re.sub(r'[^\w\s]', ' ', text)
    # Видалення зайвих пробілів
    text = re.sub(r'\s+', ' ', text)
    # Приведення до нижнього регістру
    text = text.lower().strip()
    return text


def preprocess_text(text):
    '''
    Повний NLP конвеєр обробки одного тексту.
    1. Нормалізація
    2. Токенізація (spaCy)
    3. Видалення стоп-слів
    4. Лематизація
    '''
    if nlp is None:
        return normalize_text(text)

    # Фільтрація та нормалізація
    clean_text = normalize_text(text)

    # Токенізація + видалення стоп-слів + лематизація через spaCy
    doc = nlp(clean_text)

    lemmas = []
    for token in doc:
        # Пропускаємо стоп-слова, пунктуацію та пробіли
        if token.is_stop or token.is_punct or token.is_space:
            continue
        # Пропускаємо дуже короткі токени
        if len(token.text) < 2:
            continue
        # Лематизація
        lemma = token.lemma_ if token.lemma_ != token.text else token.text
        lemmas.append(lemma)

    return ' '.join(lemmas)


def preprocess_corpus(texts):
    '''
    Застосовує NLP конвеєр до всього корпусу текстів.
    '''
    print(f'Обробка {len(texts)} текстів:')
    processed = []
    for i, text in enumerate(texts):
        processed.append(preprocess_text(text))
        if (i + 1) % 10 == 0:
            print(f'  Оброблено: {i + 1}/{len(texts)}')
    print('Обробку завершено')
    return processed


def get_token_list(text):
    '''
    Повертає список лематизованих токенів (без стоп-слів) для частотного аналізу.
    '''
    if nlp is None:
        return normalize_text(text).split()

    clean_text = normalize_text(text)
    doc = nlp(clean_text)
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
        and len(token.text) >= 2
    ]
    return tokens