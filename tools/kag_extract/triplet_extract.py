#!/usr/bin/python3

from .prompts import triplet_template

class TripletExtract(object):
  def __init__(self, tokenizer, llm, entities):
    template, parser = triplet_template(tokenizer, entities)
    self.chain = template | llm | parser
  def predict(self, query: str):
    return self.chain.invoke({'input': query})
