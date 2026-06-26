# from transformers import AutoTokenizer

# MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# text = "Patient has fever for 3 days"

# tokens = tokenizer.tokenize(text)

# print(tokens)
# from transformers import AutoTokenizer

# MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# text = "Patient has fever for 3 days"

# encoded = tokenizer(text)

# print(encoded["input_ids"])
# from transformers import AutoTokenizer, AutoModel
# import torch

# MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"

# print("Loading tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# print("Loading model...")
# model = AutoModel.from_pretrained(MODEL_NAME)

# print("Model loaded successfully!")
# from transformers import AutoTokenizer, AutoModel
# import torch

# MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModel.from_pretrained(MODEL_NAME)

# text = "Patient has fever for 3 days"

# inputs = tokenizer(text, return_tensors="pt")

# with torch.no_grad():
#     outputs = model(**inputs)

# print(outputs.last_hidden_state.shape)
# from transformers import AutoTokenizer, AutoModel
# import torch

# MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModel.from_pretrained(MODEL_NAME)

# text = "Patient has fever for 3 days"

# inputs = tokenizer(text, return_tensors="pt")

# with torch.no_grad():
#     outputs = model(**inputs)

# sentence_embedding = outputs.last_hidden_state[:, 0, :]

# #print(sentence_embedding.shape)
# print(sentence_embedding[0][:10])
from transformers import AutoTokenizer, AutoModel
import torch

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)


def get_embedding(text):

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state[:, 0, :]

    return embedding.squeeze().numpy()


result = get_embedding(
    "Patient has fever for 3 days"
)

print(type(result))
print(result.shape)