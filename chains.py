#!/usr/bin/python3

from prompts import semantic_seg_prompt

def semantic_seg_chain(llm, tokenizer, lang = 'zh'):
  assert lang in {'zh', 'en'}
  template, parser = semantic_seg_prompt(tokenizer, lang = lang)
  chain = template | llm | parser
  return chain

if __name__ == "__main__":
  from models import Qwen2
  tokenizer, llm = Qwen2(locally = True)
  chain = semantic_seg_chain(llm, tokenizer)
  with open('sem_seg.txt','r') as f:
    res = chain.invoke({'input': f.read()})
  print(res)
