#!/usr/bin/python3

from .prompts import spg_extract

class SPGExtract(object):
  def __init__(self, tokenizer, llm, schema):
    template, parser = spg_extract(tokenizer, schema)
    self.chain = template | llm | parser
  def predict(self, query: str):
    return self.chain.invoke({'input': query})
