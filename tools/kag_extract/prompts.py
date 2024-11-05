#!/usr/bin/python3

from typing import List, Dict, Optional
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

def ner_template(tokenizer, schema):
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
