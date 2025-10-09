#!/usr/bin/env python3
"""Test fine-tuned model"""

import torch
from unsloth import FastLanguageModel

print('Loading model...')
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='outputs/gpt-oss-20b-takaichi-20251009_191751/final',
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

prompts = [
    '高市早苗さんについて教えてください。',
    'サナエノミクスとは何ですか？',
]

for prompt in prompts:
    print('\n' + '='*60)
    print(f'Prompt: {prompt}')
    print('-'*60)

    messages = [{'role': 'user', 'content': prompt}]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors='pt').to('cuda')

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if '<|start|>assistant' in response:
        response = response.split('<|start|>assistant')[-1].strip()

    print(f'Response:\n{response}')
    print('='*60)
