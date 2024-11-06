#!/usr/bin/python3

import sys
import os
import unittest
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import Qwen2
from tools import load_ner_extract

class TestSPGExtract(unittest.TestCase):
  def test_function(self):
    with open(os.path.join(os.path.dirname(__file__), 'schema.json'), 'r') as f:
      schema = json.loads(f.read())
    tokenizer, llm = Qwen2(locally = True)
    ner_extract = load_ner_extract(tokenizer, llm, schema)
    with open(os.path.join(os.path.dirname(__file__), 'ner_extract.txt'), 'r') as f:
      res = ner_extract.invoke({'query': f.read()})
    print(res)

if __name__ == "__main__":
  unittest.main()
