#!/usr/bin/python3

import re
import json
import hashlib
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
    self.db = neo4j_info['db']
    with open(schema_path, 'r') as f:
      self.schema = json.loads(f.read())
    pattern = r"([^(]*)\((.*)\)"
    self.schema = {re.search(pattern, k, re.DOTALL)[1]:v for k, v in self.schema.items()}
    if tokenizer is None or llm is None:
      tokenizer, llm = Qwen2(locally = True)
    self.ner_extract = load_ner_extract(tokenizer, llm, self.schema)
    self.triplet_extract = load_triplet_extract(tokenizer, llm)
    self.entity_standard = load_entity_standard(tokenizer, llm)
  def add_entity_node(self, id, name, label, properties):
    records, summary, keys = self.driver.execute_query('merge (a: Entity {id: "$id", name: "$name", label: "$label", properties: "$properties"}) return a;', id = id, name = name, label = label, properties = properties, database_ = self.db)
  def add_official_name_edge(self, id1, id2, label):
    records, summary, keys = self.driver.execute_query('match (a {id: "$id1", label: "$label"}), (b {id: "$id2", label: "$label"}) merge (a)-[r:HAS_OFFICIAL_NAME]->(b);', id1 = id1, id2 = id2, label = label, database_ = self.db)
  def add_entity_edge(self, id1, label1, id2, label2, predicate):
    records, summary, keys = self.driver.execute_query('match (a {id: "$id1", label: "$label1"}), (b {id: "$id2", label: "$label2"}) merge (a)-[r:HAS_RELATION]->(b) set r.type = $type;', id1 = id1, label1 = label1, id2 = id2, label2 = label2, type = predicate, database_ = self.db)
  def add_entities_to_graph(self, entities):
    for entity in entities.entities:
      ent_name = entity.entity
      ent_label = entity.category
      props = entity.properties
      off_name = entity.official_name
      spg_type = self.schema[ent_label]['properties'] # Dict[str,Union[str,List[str]]]
      # add entity node
      self.add_entity_node(id = ent_name, name = ent_name, label = ent_label, properties = props)
      self.add_entity_node(id = off_name, name = off_name, label = ent_label, properties = props)
      self.add_official_name_edge(id1 = ent_name, id2 = off_name, label = ent_label)
  def add_chunk_to_graph(self, text, summary, entities):
    text_bytes = text.encode('utf-8')
    has_object = hashlib.sha256()
    has_object.update(text_bytes)
    hash_hex = has_object.hexdigest()
    records, _, keys = self.driver.execute_query('merge (a: Chunk {id: "$id", summary: "$summary", content: "$content"}) return a;', id = hash_hex, summary = summary, content = text, database_ = self.db)
    with entity in entities:
      ent_name = entity.entity
      ent_label = entity.category
      off_name = entity.official_name
      records, _, keys = self.driver.execute_query('match (a: Entity {id: "$name", label: "$category"}), (b: Chunk {id: "$hex"}) merge (a)-[r:BELONGS_TO]->(b);', name = ent_name, category = ent_label, hex = hash_hex, database_ = self.db)
  def add_edges_to_graph(self, triplets, entities):
    for triplet in triplets:
      ent1_name, predicate, ent2_name = triplet[0], triplet[1], triplet[2]
      matched1 = list(filter(lambda x: x.entity == ent1_name, entities))
      if len(matched1) != 1:
        print(f'entity {ent_name1} got multiple matches in entity list! skip triplet {triplet}!')
        continue
      ent1_label = matched1[0].category
      matched2 = list(filter(lambda x: x.entity == ent2_name, entities))
      if len(matched2) != 1:
        print(f'entity {ent_name2} got multiple matches in entity list! skip triplet {triplet}!')
        continue
      ent2_label = matched2[0].category
      self.add_entity_edge(id1 = ent1_name, label1 = ent1_label, id2 = ent2_name, label2 = ent2_label, predicate = predicate)
  def extract(self, text: str, summary: str):
    entities = self.ner_extract.invoke({'query': text})
    entities = self.entity_standard.invoke({'query': text, 'entities': str(entities)})
    self.add_entities_to_graph(entities)
    triplets = self.triplet_extract.invoke({'query': text, 'entities': str(entities)})
    self.add_edges_to_graph(triplets, entities)
    self.add_chunk_to_graph(text, summary, entities)
