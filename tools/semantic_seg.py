#!/usr/bin/python3

from pydantic import BaseModel, Field
from typing import Optional, Type, List, Dict, Union, Any
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from .semantic_seg import SemanticSegment

def load_semantic_seg(tokenizer, llm, lang = 'zh'):
  assert lang in {'zh', 'en'}
  class Schema(BaseModel):
    section_summary: str = Field(description = {
      'zh': '该小节文本的简单概括',
      'en': 'A brief summary of the section text'
    }[lang])
    section_starting_point: str = Field(description = {
      'zh': '该小节包含的原文的起点，控制在20个字左右。该分割点将被用于分割原文，因此必须可以在原文中找到！',
      'en': 'The starting point of the section in the original text, limited to about 20 characters. This segmentation point will be used to split the original text, so it must be found in the original text!'
    }[lang])
  class SemanticSegmentOutput(BaseModel):
    sections: List[Schema] = Field(description = {
      'zh': '一个Schema类型对象的list',
      'en': 'a list of Schema objects'
    }[lang])
  class SemanticSegmentInput(BaseModel):
    query: str = Field(description = "input text")
  class SemanticSegmentConfig(BaseModel):
    class Config:
      arbitrary_types_allowed = True
    predictor: SemanticSegment
  class SemanticSegmentTool(StructuredTool):
    name: str = 'semantic segmentation tool'
    description: str = 'tool segmentation according to strcutre and components of the text'
    args_schema: Type[BaseModel] = SemanticSegmentInput
    config: SemanticSegmentConfig
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> SemanticSegmentOutput:
      segments = self.config.predictor(query)
      return SemanticSegmentOutput([Schema(section_summary = segment.section_summary,
                                           section_starting_point = segment.section_starting_point) for segment in segments])
  predictor = SemanticSegment(tokenizer, llm)
  return SemanticSegmentTool(config = SemanticSegmentConfig(
    predictor = predictor
  ))
