#!/usr/bin/python3

import re
from typing import Annotated, List, Dict, Optional, Union
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

def ner_template(tokenizer, schema):
  pattern = r"([^(]*)\((.*)\)"
  schema = {re.search(pattern, k, re.DOTALL)[1]:v for k, v in schema.items()}
  schema = str(schema)
  schema = schema.replace('{','{{')
  schema = schema.replace('}','}}')
  class Entity(BaseModel):
    entity: str = Field(description = '实体文本')
    category: str = Field(description = '实体类别')
    properties: Optional[Dict[str, Union[str, List[str]]]] = Field(None, description = '实体的属性')
  class Output(BaseModel):
    entities: List[Entity] = Field(description = "Entity的list")
  parser = JsonOutputParser(pydantic_object = Output)
  instructions = parser.get_format_instructions()
  instructions = instructions.replace('{','{{')
  instructions = instructions.replace('}','}}')
  examples = [
        {
            "input": "甲状腺结节是指在甲状腺内的肿块，可随吞咽动作随甲状腺而上下移动，是临床常见的病症，可由多种病因引起。临床上有多种甲状腺疾病，如甲状腺退行性变、炎症、自身免疫以及新生物等都可以表现为结节。甲状腺结节可以单发，也可以多发，多发结节比单发结节的发病率高，但单发结节甲状腺癌的发生率较高。患者通常可以选择在普外科，甲状腺外科，内分泌科，头颈外科挂号就诊。有些患者可以触摸到自己颈部前方的结节。在大多情况下，甲状腺结节没有任何症状，甲状腺功能也是正常的。甲状腺结节进展为其它甲状腺疾病的概率只有1%。有些人会感觉到颈部疼痛、咽喉部异物感，或者存在压迫感。当甲状腺结节发生囊内自发性出血时，疼痛感会更加强烈。治疗方面，一般情况下可以用放射性碘治疗，复方碘口服液(Lugol液)等，或者服用抗甲状腺药物来抑制甲状腺激素的分泌。目前常用的抗甲状腺药物是硫脲类化合物，包括硫氧嘧啶类的丙基硫氧嘧啶(PTU)和甲基硫氧嘧啶(MTU)及咪唑类的甲硫咪唑和卡比马唑。",
            "schema": {
                "Disease": {
                    "properties": {
                        "complication": "并发症",
                        "commonSymptom": "常见症状",
                        "applicableMedicine": "适用药品",
                        "department": "就诊科室",
                        "diseaseSite": "发病部位",
                    }
                },"Medicine": {
                    "properties": {
                    }
                }
            },
            "output": [
                {
                    "entity": "甲状腺结节",
                    "category":"Disease",
                    "properties": {
                        "complication": "甲状腺癌",
                        "commonSymptom": ["颈部疼痛", "咽喉部异物感", "压迫感"],
                        "applicableMedicine": ["复方碘口服液(Lugol液)", "丙基硫氧嘧啶(PTU)", "甲基硫氧嘧啶(MTU)", "甲硫咪唑", "卡比马唑"],
                        "department": ["普外科", "甲状腺外科", "内分泌科", "头颈外科"],
                        "diseaseSite": "甲状腺",
                    }
                },{
                    "entity":"复方碘口服液(Lugol液)",
                    "category":"Medicine"
                },{
                    "entity":"丙基硫氧嘧啶(PTU)",
                    "category":"Medicine"
                },{
                    "entity":"甲基硫氧嘧啶(MTU)",
                    "category":"Medicine"
                },{
                    "entity":"甲硫咪唑",
                    "category":"Medicine"
                },{
                    "entity":"卡比马唑",
                    "category":"Medicine"
                }
            ],
        }
  ]
  examples = str(examples)
  examples = examples.replace('{','{{')
  examples = examples.replace('}','}}')
  user_message = """你是一个图谱知识抽取的专家, 基于constraint 定义的schema，从input 中抽取出所有的实体及其属性，input中未明确提及的属性返回NAN，以标准json 格式输出，结果返回list

输出格式：

%s

示范：

%s

请基于schema:

%s

抽取以下文本中的实体及属性:

{input}
"""%(instructions, examples, schema)
  messages = [
    {'role': 'user', 'content': user_message},
  ]
  prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
  template = PromptTemplate(template = prompt, input_variables = ['input'])
  return template, parser

def triplet_tetmplate(tokenizer):
  class Triplet(BaseModel):
    triplet: Annotated[List[str], 3] = Field(description = "三元组")
  class Output(BaseModel):
    triplets: List[Triplet] = Field(description = "Triplet的list")
  parser = JsonOutputParser(pydantic_object = Output)
  instructions = parser.get_format_instructions()
  instructions = instructions.replace('{','{{')
  instructions = instructions.replace('}','}}')
  examples = {
        "input": "The Rezort\nThe Rezort is a 2015 British zombie horror film directed by Steve Barker and written by Paul Gerstenberger.\n It stars Dougray Scott, Jessica De Gouw and Martin McCann.\n After humanity wins a devastating war against zombies, the few remaining undead are kept on a secure island, where they are hunted for sport.\n When something goes wrong with the island's security, the guests must face the possibility of a new outbreak.",
        "entity_list": [
            {
                "entity": "The Rezort",
                "category": "Works"
            },
            {
                "entity": "2015",
                "category": "Others"
            },
            {
                "entity": "British",
                "category": "GeographicLocation"
            },
            {
                "entity": "Steve Barker",
                "category": "Person"
            },
            {
                "entity": "Paul Gerstenberger",
                "category": "Person"
            },
            {
                "entity": "Dougray Scott",
                "category": "Person"
            },
            {
                "entity": "Jessica De Gouw",
                "category": "Person"
            },
            {
                "entity": "Martin McCann",
                "category": "Person"
            },
            {
                "entity": "zombies",
                "category": "Creature"
            },
            {
                "entity": "zombie horror film",
                "category": "Concept"
            },
            {
                "entity": "humanity",
                "category": "Concept"
            },
            {
                "entity": "secure island",
                "category": "GeographicLocation"
            }
        ],
        "output": [
            [
                "The Rezort",
                "is",
                "zombie horror film"
            ],
            [
                "The Rezort",
                "publish at",
                "2015"
            ],
            [
                "The Rezort",
                "released",
                "British"
            ],
            [
                "The Rezort",
                "is directed by",
                "Steve Barker"
            ],
            [
                "The Rezort",
                "is written by",
                "Paul Gerstenberger"
            ],
            [
                "The Rezort",
                "stars",
                "Dougray Scott"
            ],
            [
                "The Rezort",
                "stars",
                "Jessica De Gouw"
            ],
            [
                "The Rezort",
                "stars",
                "Martin McCann"
            ],
            [
                "humanity",
                "wins",
                "a devastating war against zombies"
            ],
            [
                "the few remaining undead",
                "are kept on",
                "a secure island"
            ],
            [
                "they",
                "are hunted for",
                "sport"
            ],
            [
                "something",
                "goes wrong with",
                "the island's security"
            ],
            [
                "the guests",
                "must face",
                "the possibility of a new outbreak"
            ]
        ]
  }
  examples = str(examples)
  examples = examples.replace('{','{{')
  examples = examples.replace('}','}}')
  user_message = """You are an expert specializing in carrying out open information extraction (OpenIE). Please extract any possible relations (including subject, predicate, object) from the given text, and list them following the json format {\"triples\": [[\"subject\", \"predicate\",  \"object\"]]}\n. If there are none, do not list them.\n.\n\nPay attention to the following requirements:\n- Each triple should contain at least one, but preferably two, of the named entities in the entity_list.\n- Clearly resolve pronouns to their specific names to maintain clarity.

output format:

%s

example:

%s

entity list:

{entities}

extract relations from the following text:

{input}
"""%(instructions, examples)
  messages = [
    {'role': 'user', 'content': user_message}
  ]
  prompt = tokenizer.apply_chat_template(messages, tokenize = false, add_generation_prompt = True)
  template = PromptTemplate(template = prompt, input_variables = ['input','entities'])
  return template, parser

