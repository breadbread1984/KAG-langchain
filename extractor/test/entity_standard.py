#!/usr/bin/python3

import sys
import os
import unittest
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import Qwen2
from tools import load_ner_extract, load_entity_standard

class TestEntityStandardize(unittest.TestCase):
  def test_function(self):
    # 1) named entity recognition
    with open(os.path.join(os.path.dirname(__file__), 'schema.json'), 'r') as f:
      schema = json.loads(f.read())
    tokenizer, llm = Qwen2(locally = True)
    ner_extract = load_ner_extract(tokenizer, llm, schema)
    with open(os.path.join(os.path.dirname(__file__), 'ner_extract.txt'), 'r') as f:
      entities = ner_extract.invoke({'query': f.read()})
    # 2) entity name standardize
    entities = [{'entity': entity.entity, 'category': entity.category} for entity in entities.entities]
    entities = str(entities)
    entity_standard = load_entity_standard(tokenizer, llm)
    with open(os.path.join(os.path.dirname(__file__), 'ner_extract.txt'), 'r') as f:
      entities = entity_standard.invoke({'query': f.read(), 'entities': entities})
    print(entities)

if __name__ == "__main__":
  unittest.main()

