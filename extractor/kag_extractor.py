#!/usr/bin/python3

import json
from os.path import join, exists, splitext
from .tools import load_ner_extract, load_triplet_extract, load_entity_standard
from .models import Qwen2

class KAGExtractor(object):
  def __init__(self, schema_path, tokenizer = None, llm = None):
    with open(schema_path, 'r') as f:
      self.schema = json.loads(f.read())
    if tokenizer is None or llm is None:
      tokenizer, llm = Qwen2(locally = True)
    self.ner_extract = load_ner_extract(tokenizer, llm, self.schema)
    self.triplet_extract = load_triplet_extract(tokenizer, llm)
    self.entity_standard = load_entity_standard(tokenizer, llm)
  def extract(self, text: str):
    entities = self.ner_extract.invoke({'query': text})

