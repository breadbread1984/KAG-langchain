#!/usr/bin/python3

import re
from typing import List, Dict, Optional
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
    entity: str = Field(description = "text of entity")
    type: str = Field(description = "entity type")
    category: str = Field(description = "category which entity type belongs to")
    description: str = Field(description = "a brief description of the entity")
  class Output(BaseModel):
    entities: List[Entity] = Field(description = "a list of Entity")
  parser = JsonOutputParser(pydantic_object = Output)
  instructions = parser.get_format_instructions()
  instructinos = instructions.replace('{','{{')
  instructions = instructions.replace('}','}}')
  examples = [
        {
            "input": "The Rezort\nThe Rezort is a 2015 British zombie horror film directed by Steve Barker and written by Paul Gerstenberger.\n It stars Dougray Scott, Jessica De Gouw and Martin McCann.\n After humanity wins a devastating war against zombies, the few remaining undead are kept on a secure island, where they are hunted for sport.\n When something goes wrong with the island's security, the guests must face the possibility of a new outbreak.",
            "output": [
                        {
                            "entity": "The Rezort",
                            "type": "Movie",
                            "category": "Works",
                            "description": "A 2015 British zombie horror film directed by Steve Barker and written by Paul Gerstenberger."
                        },
                        {
                            "entity": "2015",
                            "type": "Year",
                            "category": "Date",
                            "description": "The year the movie 'The Rezort' was released."
                        },
                        {
                            "entity": "British",
                            "type": "Nationality",
                            "category": "GeographicLocation",
                            "description": "Great Britain, the island that includes England, Scotland, and Wales."
                        },
                        {
                            "entity": "Steve Barker",
                            "type": "Director",
                            "category": "Person",
                            "description": "Steve Barker is an English film director and screenwriter."
                        },
                        {
                            "entity": "Paul Gerstenberger",
                            "type": "Writer",
                            "category": "Person",
                            "description": "Paul is a writer and producer, known for The Rezort (2015), Primeval (2007) and House of Anubis (2011)."
                        },
                        {
                            "entity": "Dougray Scott",
                            "type": "Actor",
                            "category": "Person",
                            "description": "Stephen Dougray Scott (born 26 November 1965) is a Scottish actor."
                        },
                        {
                            "entity": "Jessica De Gouw",
                            "type": "Actor",
                            "category": "Person",
                            "description": "Jessica Elise De Gouw (born 15 February 1988) is an Australian actress. "
                        },
                        {
                            "entity": "Martin McCann",
                            "type": "Actor",
                            "category": "Person",
                            "description": "Martin McCann is an actor from Northern Ireland. In 2020, he was listed as number 48 on The Irish Times list of Ireland's greatest film actors"
                        }
                    ]
        }
  ]
  examples = str(examples)
  examples = examples.replace('{','{{')
  examples = examples.replace('}','}}')
  user_message = """You're a very effective entity extraction system. Please extract all the entities that are important for knowledge build and question, along with type, category and a brief description of the entity. The description of the entity is based on your OWN KNOWLEDGE AND UNDERSTANDING and does not need to be limited to the context. the entity's category belongs taxonomically to one of the items defined by schema, please also output the category. Note: Type refers to a specific, well-defined classification, such as Professor, Actor, while category is a broader group or class that may contain more than one type, such as Person, Works. Return an empty list if the entity type does not exist. Please respond in the format of a JSON string.You can refer to the example for extraction.

output format:

%s

example:

%s

base on this schema:

%s

extract all the entities from the following text:

{input}
""" % (instructions, examples, schema)
  messages = [
    {'role': 'user', 'content': user_message},
  ]
  prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
  template = PromptTemplate(template = prompt, input_variables = ['input'])
  return template, parser

