#!/usr/bin/python3

from .prompts import triplet_template

class TripletExtract(object):
  def __init__(self, tokenizer, llm):
    template, parser = triplet_template(tokenizer)
    self.chain = template | llm | parser
  def predict(self, query: str, entities: str):
    return self.chain.invoke({'input': query, 'entities': entities})
