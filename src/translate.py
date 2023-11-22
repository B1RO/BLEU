from transformers import FSMTForConditionalGeneration, FSMTTokenizer
import pandas as pd
import torch
print(torch.cuda.is_available())


mname = "facebook/wmt19-en-de"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FSMTForConditionalGeneration.from_pretrained(mname).to(device)
tokenizer = FSMTTokenizer.from_pretrained(mname)


from tqdm import tqdm

df = pd.read_csv('./../translations.tsv', sep='\t', header=None, nrows=20000)
mt_translations = []

batch_size = 10
num_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)

for i in tqdm(range(num_batches)):
    start = i * batch_size
    end = start + batch_size
    inputs = df[1][start:end].tolist()
    input_ids = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    outputs = model.generate(input_ids['input_ids'])
    decoded = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    mt_translations.extend(decoded)

    # Clear up GPU memory
    del inputs
    del input_ids
    del outputs
    torch.cuda.empty_cache()

df['mt_translation'] = mt_translations
df.to_csv('translations_mt.tsv', sep='\t', index=False)
