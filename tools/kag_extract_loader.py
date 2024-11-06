#!/usr/bin/python3

from pydantic import BaseModel, Field
from typing import Annotated, Optional, Type, List, Dict, Union, Any
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from .kag_extract import NERExtract, TripletExtract, EntityStandard

def load_ner_extract(tokenizer, llm, schema):
  class Entity(BaseModel):
    entity: str = Field(description = '实体文本')
    category: str = Field(description = '实体类别')
    properties: Optional[Dict[str, Union[str,List[str]]]] = Field(None, description = '实体的属性')
  class NEROutput(BaseModel):
    entities: List[Entity] = Field(description = "Entity的list")
  class NERExtractInput(BaseModel):
    query: str = Field(description = "输入文本")
  class NERExtractConfig(BaseModel):
    class Config:
      arbitrary_types_allowed = True
    predictor: NERExtract
  class NERExtractTool(StructuredTool):
    name: str = "named entity extractor"
    description: str = "tool extracting named entities"
    args_schema: Type[BaseModel] = NERExtractInput
    config: NERExtractConfig
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> NEROutput:
      entities = self.config.predictor.predict(query)
      if type(entities) is dict: entities = entities['entities']
      return NEROutput(entities = [Entity(entity = entity['entity'], category = entity['category'], properties = entity['properties']) for entity in entities])
  predictor = NERExtract(tokenizer, llm, schema)
  return NERExtractTool(config = NERExtractConfig(
    predictor = predictor
  ))

def load_triplet_extract(tokenizer, llm):
  class Triplet(BaseModel):
    triplet: Annotated[List[str], 3] = Field(description = "三元组")
  class TripletOutput(BaseModel):
    triplets: List[Triplet] = Field(description = "Triplet的list")
  class TripletExtractInput(BaseModel):
    query: str = Field(description = "输入文本")
    entities: str = Field(description = "实体列表")
  class TripletExtractConfig(BaseModel):
    class Config:
      arbitrary_types_allowed = True
    predictor: TripletExtract
  class TripletExtractTool(StructuredTool):
    name: str = "triplet extractor"
    description: str = "tool extracting triplets"
    args_schema: Type[BaseModel] = TripletExtractInput
    config: TripletExtractConfig
    def _run(self, query: str, entities: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> TripletOutput:
      triplets = self.config.predictor.predict(query, entities)
      if type(triplets) is dict: triplets = triplets['triplets']
      return TripletOutput(triplets = [Triplet(triplet = triplet) for triplet in triplets])
  predictor = TripletExtract(tokenizer, llm)
  return TripletExtractTool(config = TripletExtractConfig(
    predictor = predictor
  ))

def load_entity_standard(tokenizer, llm):
  class Entity(BaseModel):
    entity: str = Field(description = "实体文本")
    category: str = Field(description = "实体类别")
    official_name: str = Field(description = "官方名称")
  class EntityStandardOutput(BaseModel):
    entities: List[Entity] = Field(description = "Entity的list")
  class EntityStandardInput(BaseModel):
    query: str = Field(description = "输入文本")
    entities: str = Field(description = "实体列表")
  class EntityStandardConfig(BaseModel):
    class Config:
      arbitrary_types_allowed = True
    predictor: EntityStandard
  class EntityStandardTool(StructuredTool):
    name: str = "named entity standardizer"
    description: str = "tool standardize name of entity"
    args_schema: Type[BaseModel] = EntityStandardInput
    config: EntityStandardConfig
    def _run(self, query: str, entities: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> EntityStandardOutput:
      entities = self.config.predictor.predict(query, entities)
      if type(entities) is dict: entities = entities['entities']
      return EntityStandardOutput(entities = [Entity(entity = entity['entity'], category = entity['category'], official_name = entity['official_name']) for entity in entities])
  predictor = EntityStandard(tokenizer, llm)
  return EntityStandardTool(config = EntityStandardConfig(
    predictor = predictor
  ))
