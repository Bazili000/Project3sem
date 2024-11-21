import telebot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from io import BytesIO

bot = telebot.TeleBot('8039521029:AAEp0hkIEZUXEug9mknGGlfxm8JMrcH76c8')

# Глобальные переменные для хранения датасета и модели
dataset = None
model = None
vectorizer = None

# Кнопки
markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
item1 = telebot.types.KeyboardButton("Распознать автора")
item2 = telebot.types.KeyboardButton("Точность")
item3 = telebot.types.KeyboardButton("График")
markup.add(item1, item2, item3)


# Стандартный датасет
default_data = {
    'text': [
        "Этот текст написан человеком.",
        "Я думаю, следовательно, я существую.",
        "Чувствуется легкий ветерок.",
        "Сегодня прекрасный день.",
        "Этот текст сгенерирован ИИ.",
        "Вывод модели основан на статистических данных.",
        "Вероятность следующего слова вычисляется на основе предыдущих.",
        "Анализ больших данных позволяет улучшить производительность.",

    ],
    'author': ['Человек', 'Человек', 'Человек', 'Человек', 'ИИ', 'ИИ', 'ИИ', 'ИИ']
}
def train_model():
    global dataset, model, vectorizer
    if dataset is None:
        return

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(dataset['text'])
    y = dataset['author']

    model = LogisticRegression()
    model.fit(X, y)

default_dataset = pd.DataFrame(default_data)
dataset = default_dataset # Используем датасет по умолчанию
train_model()



@bot.message_handler(commands=['start'])
def start_message(message):
    global markup
    bot.send_message(message.chat.id,
                     "Привет! Я бот, который может научиться распознавать авторов текста. Отправьте CSV файл для улучшения обучения, или используйте кнопки для работы с текущей моделью.",
                     reply_markup=markup)


@bot.message_handler(content_types=['document'])
def handle_document(message):
    global dataset, model, vectorizer
    try:
        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        # Чтение CSV файла
        dataset = pd.read_csv(BytesIO(downloaded_file))

        # Обучение модели
        train_model()

        bot.send_message(message.chat.id, "Модель обучена! Теперь можете отправлять мне текст для распознавания автора.")
    except Exception as e:
        bot.send_message(message.chat.id, f"Ошибка при обработке файла: {e}")


@bot.message_handler(func=lambda message: True, content_types=['text'])
def handle_text(message):
    global model, vectorizer, markup
    if message.text == "Распознать автора":
        bot.send_message(message.chat.id, "Введите текст для распознавания:", reply_markup=telebot.types.ForceReply())
    elif message.text == "Точность":
        accuracy_check(message)
    elif message.text == "График":
        plot_accuracy(message)
    else:
        if model is None:
            bot.send_message(message.chat.id, "Сначала нужно обучить модель. Отправьте мне CSV файл с данными.")
            return

        text = [message.text]
        X_test = vectorizer.transform(text)
        prediction = model.predict(X_test)[0]
        bot.send_message(message.chat.id, f"Предполагаемый автор: {prediction}", reply_markup=markup)

@bot.message_handler(commands=['accuracy'])
def accuracy_check(message):
    global dataset, model, vectorizer

    if model is None or dataset is None:
        bot.reply_to(message, "Модель не обучена или датасет отсутствует.")
        return

    X = vectorizer.transform(dataset['text'])
    y = dataset['author']
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)

    bot.reply_to(message, f"Точность на обучающем датасете: {accuracy:.2f}")

@bot.message_handler(commands=['plot'])
def plot_accuracy(message):
    global dataset, model, vectorizer
    if model is None or dataset is None:
        bot.reply_to(message, "Модель не обучена или датасет отсутствует.")
        return

    accuracies = []
    dataset_sizes = range(10, len(dataset), 10) # Проверяем точность на подмножествах данных

    for size in dataset_sizes:
        subset = dataset.sample(size)
        X_subset = vectorizer.transform(subset['text'])
        y_subset = subset['author']
        predictions = model.predict(X_subset)
        accuracy = np.mean(predictions == y_subset)
        accuracies.append(accuracy)

    plt.plot(dataset_sizes, accuracies)
    plt.xlabel("Размер датасета")
    plt.ylabel("Точность")
    plt.title("Зависимость точности от размера датасета")

    # Сохраняем график в буфер
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Отправляем график пользователю
    bot.send_photo(message.chat.id, buf)
    plt.clf() # Очищаем график


bot.polling()