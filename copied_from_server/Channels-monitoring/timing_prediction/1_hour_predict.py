import torch
from transformers import DistilBertModel, DistilBertTokenizer
import pandas as pd
import torch.nn as nn
from datetime import timedelta

reaction_types = ['☃', '⚡', '✍', '❤', '❤‍🔥', '🆒', '🌚', '🌭', '🍌', '🍓', '🍾', '🎃', '🎅', '🎉', '🏆', '🐳', '👀', '👌',
                  '👍', '👎', '👏', '👨‍💻', '👻', '👾', '💅', '💊', '💋', '💘', '💩', '🔥', '🕊', '🖕', '🗿', '😁', '😈', '😍',
                  '😎', '😐', '😡', '😢', '😨', '😭', '😱', '😴', '🙈', '🙉', '🙊', '🙏', '🤓', '🤔', '🤗', '🤡', '🤣', '🤨',
                  '🤩', '🤪', '🤬', '🤮', '🤯', '🤷', '🤷‍♀', '🤷‍♂', '🥰', '🥱', '🥴', '🦄', '🫡']


class ReactionPredictor(nn.Module):
    def __init__(self):
        super(ReactionPredictor, self).__init__()
        self.fc = nn.Linear(768, len(reaction_types))

    def forward(self, x):
        return self.fc(x)


model = ReactionPredictor()
model.load_state_dict(torch.load("reaction_predictor_model.pth"))
model.eval()

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")

cleaned_data = pd.read_csv("cleaned_data.csv")

cleaned_data['Date'] = pd.to_datetime(cleaned_data['Date'])


def predict_hourly_reactions(post_id):
    post_data = cleaned_data[cleaned_data['ID'] == post_id].sort_values(by='Date')
    if post_data.empty:
        print(f"Пост с ID {post_id} не найден.")
        return

    first_entry = post_data.iloc[0]
    post_text = first_entry['Text']
    time_one_hour_later = first_entry['Date'] + timedelta(hours=1)

    hourly_reactions_data = post_data[post_data['Date'] <= time_one_hour_later]
    final_reactions = hourly_reactions_data.iloc[-1] if not hourly_reactions_data.empty else first_entry

    actual_reactions = eval(final_reactions['Reactions'])

    inputs = tokenizer(post_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        embeddings = distilbert(**inputs).last_hidden_state[:, 0, :]
        predicted_reactions = model(embeddings).squeeze().tolist()

    for i, emoji in enumerate(reaction_types):
        predicted_count = max(0, round(predicted_reactions[i]))
        actual_count = actual_reactions.get(emoji, 0)
        print(f"{emoji}: {predicted_count}, {actual_count}")


post_id = 27788
predict_hourly_reactions(post_id)
