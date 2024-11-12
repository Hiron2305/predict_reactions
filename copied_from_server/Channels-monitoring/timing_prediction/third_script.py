import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Загрузка данных и эмбеддингов")
df = pd.read_csv("cleaned_data.csv")
embeddings = torch.load("../embeddings.pt")

logging.info("Применение эмбеддингов к текстам")
df['embedding'] = df['Text'].apply(lambda text: embeddings.get(text))
df = df.dropna(subset=['embedding'])

df = df[df['Reactions'].notnull()]
reaction_types = set()
for reactions in df['Reactions']:
    try:
        parsed_reactions = eval(reactions) if isinstance(reactions, str) and reactions.startswith("{") else {}
        if isinstance(parsed_reactions, dict):
            reaction_types.update(parsed_reactions.keys())
    except (SyntaxError, TypeError, ValueError) as e:
        logging.warning(f"Ошибка при обработке реакции: {reactions} - {e}")

reaction_types = sorted(filter(lambda x: x is not None, reaction_types))
emoji_to_id = {emoji: idx for idx, emoji in enumerate(reaction_types)}
logging.info(f"Уникальные типы реакций (emoji): {reaction_types}")


def reaction_vector(reactions):
    vec = [0] * len(reaction_types)
    if isinstance(reactions, str):
        try:
            reactions = eval(reactions)
        except (SyntaxError, TypeError, ValueError) as e:
            logging.warning(f"Ошибка при преобразовании реакции: {reactions} - {e}")
            reactions = {}
    if isinstance(reactions, dict):
        for emoji, count in reactions.items():
            if emoji in emoji_to_id:
                vec[emoji_to_id[emoji]] = count
    return vec


df['reaction_vector'] = df['Reactions'].apply(reaction_vector)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


class ReactionDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        embedding = torch.tensor(self.data.iloc[idx]['embedding'], dtype=torch.float32)
        reaction_vector = torch.tensor(self.data.iloc[idx]['reaction_vector'], dtype=torch.float32)
        return embedding, reaction_vector


train_dataset = ReactionDataset(train_df)
test_dataset = ReactionDataset(test_df)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)


class ReactionPredictor(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(ReactionPredictor, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)


model = ReactionPredictor(input_size=len(train_df['embedding'].iloc[0]), output_size=len(reaction_types))
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

logging.info("Начало обучения модели")
num_epochs = 10
initial_loss = None

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    batch_count = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batch_count += 1

    avg_loss = epoch_loss / batch_count

    if initial_loss is None:
        initial_loss = avg_loss

    loss_percentage = (avg_loss / initial_loss) * 100 if initial_loss else 100

    logging.info(f"Эпоха [{epoch + 1}/{num_epochs}], Потеря: {avg_loss:.4f} ({loss_percentage:.2f}% от начальной)")

model_save_path = "../reaction_predictor_model.pth"
torch.save(model.state_dict(), model_save_path)
logging.info(f"Модель сохранена в {model_save_path}")

logging.info("Обучение завершено")
