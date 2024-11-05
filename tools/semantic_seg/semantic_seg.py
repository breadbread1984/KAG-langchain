#!/usr/bin/python3

from .prompts import semantic_seg_prompt

class SemanticSegment(object):
  def __init__(self, tokenizer, llm, lang):
    template, parser = semantic_seg_prompt(tokenizer, lang)
    self.chain = template | llm | parser
  def predict(self, query: str):
    return self.chain.invoke({'input': query})
