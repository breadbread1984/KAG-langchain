#!/usr/bin/python3

from pydantic import BaseModel, Field
from typing import Optional, Type, List, Dict, Union, Any
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from .kag_extract import NERExtract

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
    description: str = "tool extract named entity"
    args_schema: Type[BaseModel] = NERExtractInput
    config: NERExtractConfig
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> NEROutput:
      entities = self.config.predictor.predict(query)
      return NEROutput(entities = [Entity(entity = entity['entity'], category = entity['category'], properties = entity['properties']) for entity in entities])
  predictor = NERExtract(tokenizer, llm, schema)
  return NERExtractTool(config = NERExtractConfig(
    predictor = predictor
  ))
