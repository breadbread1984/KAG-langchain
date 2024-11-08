#!/usr/bin/python3

import sys
import os
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from extractor import SemanticSegmentExtractor, Qwen2

class TestSemanticSegExtractor(unittest.TestCase):
  def test_function(self):
    tokenizer, llm = Qwen2(locally = True)
    extractor = SemanticSegmentExtractor(tokenizer, llm)
    file_path = os.path.join(os.path.dirname(__file__), 'sem_seg.txt')
    segments = extractor.extract(file_path)
    print(segments)

if __name__ == "__main__":
  unittest.main()
