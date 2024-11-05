#!/usr/bin/python3

from .prompts import ner_template

class NERExtract(object):
  def __init__(self, tokenizer, llm, schema):
    template, parser = ner_template(tokenizer, schema)
    self.chain = template | llm | parser
  def predict(self, query: str):
    return self.chain.invoke({'input': query})

