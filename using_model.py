import torch
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
import json
import torch.nn as nn

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

model = LinearRegressionModel(input_dim=771).to(DEVICE)  
model.load_state_dict(torch.load('model.pth', map_location=DEVICE))
model.eval()

with open('emoji_mapping.json', 'r', encoding='utf-8') as f:
    emoji_to_id = json.load(f)
    id_to_emoji = {v: k for k, v in emoji_to_id.items()}

def preprocess_text_bert(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = DistilBertModel.from_pretrained('distilbert-base-uncased')(**inputs)
    result = outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()
    
    padding = np.zeros(771 - len(result))
    result_padded = np.concatenate([result, padding])
    
    return result_padded

def predict_reactions(text):
    embedding = preprocess_text_bert(text)
    
    embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        predictions = model(embedding_tensor).cpu().numpy().flatten()
    
    predictions = np.maximum(predictions, 0)

    pred_dict = {id_to_emoji.get(i, f"Unknown_{i}"): count for i, count in enumerate(predictions)}

    return pred_dict

def save_predictions_to_csv(hourly_pred, daily_pred, weekly_pred, filename="predictions.csv"):
    data = {
        "Hourly Predictions": [],
        "Daily Predictions": [],
        "Weekly Predictions": []
    }

    for hour, day, week in zip(hourly_pred.items(), daily_pred.items(), weekly_pred.items()):
        hour_emoji, hour_count = hour
        day_emoji, day_count = day
        week_emoji, week_count = week
        data["Hourly Predictions"].append(f"{hour_emoji}: {hour_count:.2f}")
        data["Daily Predictions"].append(f"{day_emoji}: {day_count:.2f}")
        data["Weekly Predictions"].append(f"{week_emoji}: {week_count:.2f}")

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"Predictions saved to {filename}")

def display_predictions(text):
    pred_hourly = predict_reactions(text)
    pred_daily = predict_reactions(text)
    pred_weekly = predict_reactions(text)

    save_predictions_to_csv(pred_hourly, pred_daily, pred_weekly)

if __name__ == '__main__':
    sample_text = "Главное за ночь и утро: НАСТОЯЩИЙ МАТЕРИАЛ (ИНФОРМАЦИЯ) ПРОИЗВЕДЕН И РАСПРОСТРАНЕН ИНОСТРАННЫМ АГЕНТОМ THE BELL ЛИБО КАСАЕТСЯ ДЕЯТЕЛЬНОСТИ ИНОСТРАННОГО АГЕНТА THE BELL. 18+ — Признание Дональда Трампа виновным по уголовному делу не означает, что его поражение на президентских выборах 5 ноября предопределено. Вероятность тюремного заключения мала, при этом часть финансовых элит США продолжает считать Трампа лучшим кандидатом, чем Джо Байден. Более того, экс-президент укрепляет свои позиции на Уолл-стрит, пишет Bloomberg. — ВСУ провели новую комбинированную массированную атаку беспилотниками по территории России, Минобороны отчиталось о сбитых ночью 29 беспилотниках и пяти ПКР «Нептун» над Краснодарским краем. Подтвержден пожар на нефтебазе в Темрюкском районе. Утром появились сообщения о беспилотниках над Татарстаном, закрытии аэропортов Казани и Нижнекамска, а также эвакуации на «Казаньоргсинтезе». — Банки уже почти не сомневаются в решении ЦБ по ставке 7 июня. Главный банк страны — государственный Сбербанк — с сегодняшнего дня поднял максимальную ставку по вкладам до 18% годовых. Самый крупный частный Альфа-банк — до 17%. — Минфин продолжает подводить базу под прогрессивную шкалу НДФЛ. Министерство представило данные о налогоплательщиках с годовым доходом выше 10 млн рублей, которым предстоит платить налог от 18% вместо 15%. Оказалось, что таких всего 167 тысяч человек (0,3%), но в прошлом году они выплатили государству почти пятую часть (19% или 1,2 трлн рублей) НДФЛ.— В Совфеде разработан законопроект «Об охране голоса» — в том числе сгенерированного нейросетями. По сути он повторяет законодательство об изображениях: использование голоса гражданина допускается только с его согласия, а после смерти — с согласия детей, супруга или родителей. Согласие на посмертную «жизнь» голоса, как и с фото, не потребуется для использования в государственных или общественных интересах, если запись производилась за плату или в публичных местах.— Складские мощности в России смещаются с северо-запада европейской части поближе к главному торговому партнеру - Китаю. Central Properties Сергея Егорова и Дениса Степанова приобрела крупные склады в Красноярске за 4,5–5 млрд рублей, одним из арендаторов станет Ozon."
    display_predictions(sample_text)
