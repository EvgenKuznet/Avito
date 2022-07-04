from typing import Tuple, Union
import pandas as pd
import catboost as cb
import pymorphy2
from nltk import tokenize
import re
import pickle as pkl
import os

morph = pymorphy2.MorphAnalyzer(lang='ru')


def man_proc(text):
    tokens = tokenize.word_tokenize(text)

    dig_count = 0  # pure numeric
    ru_count = 0
    en_count = 0
    has_digit = 0
    ru_dict_count = 0

    for token in tokens:
        if token.isnumeric():
            dig_count += 1
        if bool(re.search('[а-яА-Я]', token)):
            ru_count += 1
        if bool(re.search('[a-zA-Z]', token)):
            en_count += 1
        if any(char.isdigit() for char in token):
            has_digit += 1
        if morph.word_is_known(token):
            ru_dict_count += 1

    return len(tokens), dig_count, ru_count, en_count, has_digit, ru_dict_count


def digits_count(text):
    digits = [0] * 13
    for char in text:
        for i in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            if char == i:
                digits[int(i)] += 1
    text = text.replace("/n", "").replace("/t", "")
    for char in text:
        for ind, special in enumerate(['-', "/", "@"]):
            if char == special:
                digits[ind] += 1
    return digits


def find_duplicate_words(data):
    title = data[0]
    descr = data[1]
    title_tokens = tokenize.word_tokenize(title)
    descr_tokens = tokenize.word_tokenize(descr)
    return len(set(title_tokens) & set(descr_tokens))


def join(source_dir, dest_file, read_size):
    # Create a new destination file
    output_file = open(dest_file, 'wb')

    # Get a list of the file parts
    parts = os.listdir(source_dir)

    # Go through each portion one by one
    for file in parts:

        # Assemble the full path to the file
        path = source_dir+'/'+file

        # Open the part
        input_file = open(path, 'rb')

        while True:
            # Read all bytes of the part
            bytes = input_file.read(read_size)

            # Break out of loop if we are at end of file
            if not bytes:
                break

            # Write the bytes to the output file
            output_file.write(bytes)

        # Close the input file
        input_file.close()

    # Close the output file
    output_file.close()


def task1(df: pd.DataFrame) -> float:
    df.datetime_submitted = pd.to_datetime(df.datetime_submitted)
    df['hour'] = df.datetime_submitted.apply(lambda x: x.hour)
    print("Restore descr_vectorizer")
    join(source_dir='./lib/descr_vec', dest_file="./lib/descr_vectorizer.pkl", read_size=90000000)

    df = pd.concat((df, pd.DataFrame(list(df.description.apply(man_proc)),
                                     columns=['descr_len', 'descr_dig_count', 'descr_ru_count', 'descr_en_count',
                                              'descr_has_digit_count', 'descr_ru_dict_count'])), axis=1)

    print ("Manual processing")
    df = pd.concat((df, pd.DataFrame(list(df.title.apply(man_proc)),
                                     columns=['title_len', 'title_dig_count', 'title_ru_count', 'title_en_count',
                                              'title_has_digit_count', 'title_ru_dict_count'])), axis=1)

    df = pd.concat((df, pd.DataFrame(list(df.description.apply(digits_count)),
                                     columns=[f"count_{i}" for i in range(13)])), axis=1)
    df['title_descr_duplicates'] = df[['title', 'description']].apply(find_duplicate_words, axis=1)

    print("descr_vectorizer")
    with open('lib/descr_vectorizer.pkl', 'rb') as f:
        descr_vectorizer = pkl.load(f)
    X = descr_vectorizer.transform(df.description)
    descr_df = pd.DataFrame(X.toarray(), columns=[f"descr_{i}" for i in range(X.shape[1])])
    df = pd.concat((df, descr_df), axis=1)

    print("title_vectorizer")
    with open('lib/title_vectorizer.pkl', 'rb') as f:
        title_vectorizer = pkl.load(f)
    X = title_vectorizer.transform(df.description)
    title_df = pd.DataFrame(X.toarray(), columns=[f"title_{i}" for i in range(X.shape[1])])
    df = pd.concat((df, title_df), axis=1)

    X_val = df.drop(['title', 'description', 'datetime_submitted'], axis=1)
    if 'is_bad' in X_val.columns:
        X_val.drop(['is_bad'], axis=1)
    #y_val = df['is_bad']

    cb_clf = cb.CatBoostClassifier()
    cb_clf.load_model("lib/Avito_cb_model.sav")
    preds = cb_clf.predict_proba(X_val)[:, 1]
    return preds


def task2(description: str) -> Union[Tuple[int, int], Tuple[None, None]]:
    description_size = len(description)
    if description_size % 2 == 0:
        return None, None
    else:
        return 0, description_size
