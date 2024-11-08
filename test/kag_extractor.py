#!/usr/bin/python3

import sys
import os
import unittest
from absl import app, flags

FLAGS = flags.FLAGS

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from extractor import SemanticSegmentExtractor, KAGExtractor

def add_options():
  flags.DEFINE_string("schema", default = os.path.join(os.path.dirname(__file__), "schema.json"), help = 'path to schema json file')
  flags.DEFINE_string("neo4j_host", default = "bolt://localhost:7687", help = 'host of neo4j')
  flags.DEFINE_string("neo4j_user", default = "neo4j", help = 'user name of neo4j')
  flags.DEFINE_string("neo4j_password", default = "neo4j", help = "password of neo4j")
  flags.DEFINE_string("neo4j_db", default = "neo4j", help = "which database to use")

class TestKAGExtractor(unittest.TestCase):
  def test_function(self):
      extractor = SemanticSegmentExtractor()
      file_path = os.path.join(os.path.dirname(__file__), 'kag_extractor.txt')
      segments = extractor.extract(file_path)
      extractor = KAGExtractor(FLAGS.schema,
                               neo4j_info = {
                                 'host': FLAGS.neo4j_host,
                                 'user': FLAGS.neo4j_user,
                                 'password': FLAGS.neo4j_password,
                                 'db': FLAGS.neo4j_db
                               })
      for segment in segments:
        summary, text = segment['summary'], segment['text']
        extractor.extract(text, summary)

def main(argv):
  argv = argv[:1] + argv[2:]
  unittest.main(argv = argv)

if __name__ == "__main__":
  add_options()
  app.run(main)
