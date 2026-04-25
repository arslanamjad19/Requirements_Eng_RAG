# download_model.py
from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-en-de"
local_dir = "./opus-mt-en-de-local"

tokenizer = MarianTokenizer.from_pretrained(model_name)
model     = MarianMTModel.from_pretrained(model_name)

tokenizer.save_pretrained(local_dir)
model.save_pretrained(local_dir)
print("Model saved to", local_dir)

model_gname = "Helsinki-NLP/opus-mt-de-en"
local_gdir = "./opus-mt-de-en-local"

gtokenizer = MarianTokenizer.from_pretrained(model_gname)
gmodel     = MarianMTModel.from_pretrained(model_gname)

gtokenizer.save_pretrained(local_gdir)
gmodel.save_pretrained(local_gdir)
print("Model saved to", local_gdir)


# From english to german
# import os
# import time
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# from transformers import pipeline
# local_dir = "./opus-mt-en-de-local"
# translator = pipeline("translation", model=local_dir, tokenizer=local_dir, device=0)  # device=0 for GPU

# english_texts = [
#     "The vehicle shall allow drivers to customize their preferred charging station filters, such as distance, type (e.g., DC Fast Charger, Level 2), and availability."
# ]
# t0 = time.time()
# results = translator(english_texts, max_length=128, batch_size=16)
# t1 = time.time()
# print("Total time:", t1-t0, "per call:", (t1-t0)/len(results))
# german = [r["translation_text"] for r in results]
# print(german)

# translate_pipeline_offline.py
# import os
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# from transformers import pipeline
# local_dir = "./opus-mt-de-en-local"
# translator = pipeline("translation", model=local_dir, tokenizer=local_dir, device=0)  # device=0 for GPU

# german_texts = [
#     "Automatisches Feststellen der Spannungsfreiheit (AFES)", "Hausstromwächter", "Ruhestrommanagement"
# ]
# results = translator(german_texts, max_length=128, batch_size=16)
# english = [r["translation_text"] for r in results]
# print(english)


