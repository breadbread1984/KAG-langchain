#!/usr/bin/python3

from fuzzywuzzy import process
from os.path import join, exists, splitext
from langchain_community.document_loaders import UnstructuredPDFLoader, TextLoader, UnstructuredMarkdownLoader
from .tools import load_semantic_seg
from .models import Qwen2

class SemanticSegmentExtractor(object):
  def __init__(self, tokenizer = None, llm = None):
    if tokenizer is None or llm is None:
      tokenizer, llm = Qwen2(locally = True)
    self.semantic_seg = load_semantic_seg(tokenizer, llm, 'zh')
  def extract(self, file_path: str):
    stem, ext = splitext(file_path)
    if ext.lower() == '.md':
      loader = UnstructuredMarkdownLoader(file_path, model = 'single', strategy = 'fast')
    elif ext.lower() == '.txt':
      loader = TextLoader(file_path)
    elif ext.lower() == '.pdf':
      loader = UnstructuredPDFLoader(file_path, model = 'single', strategy = 'fast')
    else:
      raise Exception('unknown file type!')
    text = ' '.join([doc.page_content for doc in loader.load()])
    sections = self.semantic_seg.invoke({'query': text})
    segments = list()
    beg = 0
    section_summary = None
    for idx, section in enumerate(sections.sections):
      pos = text.find(section.section_starting_point, beg)
      if pos < 0:
        matches = process.extract(text, section.section_starting_point, limit = 3)
        matches = list(filter(lambda x: x[0] == section.section_starting_point[0], matches))
        if len(matches) == 0:
          raise Exception('unmatched section start string!')
        pos = matches[0][1]
      if idx != 0: segments.append({'summary': section_summary, 'text': text[beg:pos]})
      beg = pos
      section_summary = section.section_summary
    segments.append({'summary': section_summary,'text': text[beg:]})
    return segments
