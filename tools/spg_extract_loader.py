#!/usr/bin/python3

from pydantic import BaseModel, Field
from typing import Optional, Type, List, Dict, Union, Any
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from .spg_extract import SPGExtract

def load_spg_extract(tokenizer, llm, schema):
  class Entity(BaseModel):
    entity: str = Field(description = '实体文本')
    category: str = Field(description = '实体类别')
    properties: Optional[Dict[str, str]] = Field(None, description = '实体的属性')
  class SPGExtractOutput(BaseModel):
    entities: List[Entity] = Field(description = "Entity的list")
  class SPGExtractInput(BaseModel):
    query: str = Field(description = "输入文本")
  class SPGExtractConfig(BaseModel):
    class Config:
      arbitrary_types_allowed = True
    predictor: SPGExtract
  class SPGExtractTool(StructuredTool):
    name: str = "semantic-enhanced programmable graph extractor"
    description: str = 'tool extract entity and its properties'
    args_schema: Type[BaseModel] = SPGExtractInput
    config: SPGExtractConfig
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> SPGExtractOutput:
      entities = self.config.predictor.predict(query)
      return SPGExtractOutput(entities = [Entity(entity = entity['entity'], category = entity['category'], properties = entity['properties']) for entity in entities])
  predictor = SPGExtract(tokenizer, llm, schema)
  return SPGExtractTool(config = SPGExtractConfig(
    predictor = predictor
  ))

