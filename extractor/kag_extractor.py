#!/usr/bin/python3

import re
import json
from os.path import join, exists, splitext
from neo4j import GraphDatabase
from .tools import load_ner_extract, load_triplet_extract, load_entity_standard
from .models import Qwen2

class KAGExtractor(object):
  def __init__(self, schema_path, tokenizer = None, llm = None,
               neo4j_info = {
                 'host': 'bolt://localhost:7687',
                 'user': 'neo4j',
                 'password': 'neo4j',
                 'db': 'neo4j'
               }):
    self.driver = GraphDatabase.driver(neo4j_info['host'], auth = (neo4j_info['user'], neo4j_info['password']))
    self.db = db
    with open(schema_path, 'r') as f:
      self.schema = json.loads(f.read())
    pattern = r"([^(]*)\((.*)\)"
    self.schema = {re.search(pattern, k, re.DOTALL)[1]:v for k, v in self.schema.items()}
    if tokenizer is None or llm is None:
      tokenizer, llm = Qwen2(locally = True)
    self.ner_extract = load_ner_extract(tokenizer, llm, self.schema)
    self.triplet_extract = load_triplet_extract(tokenizer, llm)
    self.entity_standard = load_entity_standard(tokenizer, llm)
  def add_property_node(self, id, value, label):
    records, summary, keys = self.driver.execute_query('merge (a: Property {id: "$id", value: "$value", label: "$label"}) return a;', id = id, value = value, label = label, database_ = self.db)
  def add_entity_node(self, id, name, label):
    records, summary, keys = self.driver.execute_query('merge (a: Entity {id: "$id", name: "$name", label: "$label"}) return a;', id = id, name = name, label = label, database_ = self.db)
  def add_edge(self, id1, label1, id2, label2, prop_name):
    records, summary, keys = self.driver.execute_query('match (a {id: "$id1", label: "$label1"}), (b {id: "$id2", label: "$label2"}) merge (a)-[:HAS_PROPERTY]->(b) set r.property = $property;', id1 = id1, label1 = label1, id2 = id2, label2 = label2, property = prop_name, database_ = self.db)
  def add_entities_to_graph(self, entities):
    for entity in entities.entities:
      ent_name = entity.entity
      ent_label = entity.category
      props = entity.properties
      spg_type = self.schema[ent_label]['properties'] # Dict[str,Union[str,List[str]]]
      # add entity node
      self.add_entity_node(id = ent_name, name = ent_name, label = ent_label)
      for prop_name, prop_value in props.items():
        # skip invalid value
        if prop_value == "NAN": continue
        # skip unknown property
        if prop_name not in spg_type: continue
        prop_value = prop_value if isinstance(prop_value, list) else [prop_value]
        for v in prop_value:
          # add property node
          self.add_property_node(id = v, value = v, label = prop_name)
          # add edge between entity node and property node
          self.add_edge(id = ent_name, name = ent_name, label = ent_label, prop_name)      
  def extract(self, text: str):
    entities = self.ner_extract.invoke({'query': text})
    self.add_entities_to_graph(entities)
    triplets = self.triplet_extract.invoke({'query': text, 'entities': str(entities)})
    entities = self.entity_standard.invoke({'query': text, 'entities': str(entities)})
    
