#!/usr/bin/python3

import regex as re
from os.path import join, exists, splitext
from langchain_community.document_loaders import UnstructuredPDFLoader, TextLoader, UnstructuredMarkdownLoader
from .tools import load_semantic_seg
from .models import Qwen2

class SemanticSegmentExtractor(object):
  def __init__(self, tokenizer = None, llm = None):
    if tokenizer is None or llm is None:
      tokenizer, llm = Qwen2(locally = True)
    self.semantic_seg = load_semantic_seg(tokenizer, llm, 'zh')
  def fuzzy_find_with_first_char_match(self, text, query, max_l_dist=3):
    # 确保第一个字符匹配
    first_char = query[0]
    pattern = f"(?<={first_char})({query[1:]}){{e<={max_l_dist}}}"
    matches = list(re.finditer(pattern, text))
    # 过滤匹配以确保第一个字符相等
    filtered_matches = [
        (text[match.start()-1:match.end()], match.start() - 1, match.end())
        for match in matches
        if match.start() > 0 and text[match.start() - 1] == first_char
    ]
    return filtered_matches
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
        matches = self.fuzzy_find_with_first_char_match(text, section.section_starting_point)
        matches = list(filter(lambda x: x[1] >= beg, matches))
        if len(matches) == 0:
          raise Exception('unmatched section start string!')
        pos = matches[0][1]
      if idx != 0: segments.append({'summary': section_summary, 'text': text[beg:pos]})
      beg = pos
      section_summary = section.section_summary
    segments.append({'summary': section_summary,'text': text[beg:]})
    return segments
