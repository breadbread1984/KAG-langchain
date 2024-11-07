#!/usr/bin/python3

from .prompts import entity_standard_template

class EntityStandard(object):
  def __init__(self, tokenizer, llm):
    template, parser = entity_standard_template(tokenizer)
    self.chain = template | llm | parser
  def predict(self, query: str, entities: str):
    return self.chain.invoke({'input': query, 'entities': entities})
