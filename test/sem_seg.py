#!/usr/bin/python3

import sys
import os
import unittest
from models import Qwen2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools import load_semantic_seg

class TestSemanticSeg(unittest.TestCase):
  def test_function(self):
    tokenizer, llm = Qwen2(locally = True)
    semantic_seg = load_semantic_seg(tokenizer, llm, 'zh')
    with open('sem_seg.txt', 'r') as f:
      res = semantic_seg.invoke({'input': f.read()})
    print(res)

if __name__ == "__main__":
  unittest.main()
