# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
# Copyright (c) 2024 Jiashuo Sun.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified by Graphcore Ltd.

# Adapted from the prompts in https://github.com/GasolSun36/ToG/blob/main/ToG/prompt_list.py


def wiki_relation_prompt(num_candidates: int = 3):
    return (
        f"Please retrieve {num_candidates} relations that contribute to the question and rate their contribution on a scale from 0 to 1 (the sum of the scores of {num_candidates} relations is 1). Respond only with the relations (separated by semicolon) and their scores.\n"
        "Question: Mesih Pasha's uncle became emperor in what year?\n"
        "Topic Entity: Mesih Pasha\n"
        "Relations to choose from:\n"
        "1. child\n"
        "2. country of citizenship\n"
        "3. date of birth\n"
        "4. family\n"
        "5. father\n"
        "6. languages spoken, written or signed\n"
        "7. military rank\n"
        "8. occupation\n"
        "9. place of death\n"
        "10. position held\n"
        "11. religion or worldview\n"
        "12. sex or gender\n"
        "13. sibling\n"
        "14. significant event\n"
        "Relevant relations: family (Score: 0.5);father (Score: 0.4);position held (Score: 0.1)\n\n"
        # "Relevant relations: 1. {{family (Score: 0.5)}}: This relation is highly relevant as it can provide information about the family background of Mesih Pasha, including his uncle who became emperor.\n"
        # "2. {{father (Score: 0.4)}}: Uncle is father's brother, so father might provide some information as well.\n"
        # "3. {{position held (Score: 0.1)}}: This relation is moderately relevant as it can provide information about any significant positions held by Mesih Pasha or his uncle that could be related to becoming an emperor.\n\n"
        "Question: Van Andel Institute was founded in part by what American businessman, who was best known as co-founder of the Amway Corporation?\n"
        "Topic Entity: Van Andel Institute\n"
        "Relations to choose from:\n"
        "1. affiliation\n"
        "2. country\n"
        "3. donations\n"
        "4. educated at\n"
        "5. employer\n"
        "6. headquarters location\n"
        "7. legal form\n"
        "8. located in the administrative territorial entity\n"
        "9. total revenue\n"
        "Relevant relations: affiliation (Score: 0.4);donations (Score: 0.3);educated at (Score: 0.3)\n\n"
        # "Relevant relations: 1. {{affiliation (Score: 0.4)}}: This relation is relevant because it can provide information about the individuals or organizations associated with the Van Andel Institute, including the American businessman who co-founded the Amway Corporation.\n"
        # "2. {{donations (Score: 0.3)}}: This relation is relevant because it can provide information about the financial contributions made to the Van Andel Institute, which may include donations from the American businessman in question.\n"
        # "3. {{educated at (Score: 0.3)}}: This relation is relevant because it can provide information about the educational background of the American businessman, which may have influenced his involvement in founding the Van Andel Institute.\n\n"
    )


def score_entity_prompt():
    return (
        "Please score the entities' contribution to the question on a scale from 0 to 1 (the sum of the scores of all entities is 1).\n"
        'Question: Staten Island Summer, starred what actress who was a cast member of "Saturday Night Live"?\n'
        "Relation: cast member\n"
        "Entites: Ashley Greene; Bobby Moynihan; Camille Saviola; Cecily Strong; Colin Jost; Fred Armisen; Gina Gershon; Graham Phillips; Hassan Johnson; Jackson Nicoll; Jim Gaffigan; John DeLuca; Kate Walsh; Mary Birdsong\n"
        # 'Scores: 0.0, 0.0, 0.0, 0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0\n'
        "Scores: to score the entities' contribution to the question, we need to determine which entities are relevant to the question and have a higher likelihood of being the correct answer.\n"
        'In this case, we are looking for an actress who was a cast member of "Saturday Night Live" and starred in the movie "Staten Island Summer." Based on this information, we can eliminate entities that are not actresses or were not cast members of "Saturday Night Live."\n'
        'The relevant entities that meet these criteria are:\n- Ashley Greene\n- Cecily Strong\n- Fred Armisen\n- Gina Gershon\n- Kate Walsh\n\nTo distribute the scores, we can assign a higher score to entities that are more likely to be the correct answer. In this case, the most likely answer would be an actress who was a cast member of "Saturday Night Live" around the time the movie was released.\n'
        "Based on this reasoning, the scores could be assigned as follows:\n- Ashley Greene: 0\n- Cecily Strong: 0.4\n- Fred Armisen: 0.2\n- Gina Gershon: 0\n- Kate Walsh: 0.4\n\n"
    )


def evaluate_complete_prompt():
    return (
        "Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are first asked to answer whether the retrieved knowledge contains the information necessary to answer the question ({Yes} or {No}).\n"
        "If the answer is {Yes}, you are then asked to provide the answer to the original question. You can only use the information contained in the triplets to produce the answer, and don't use any other knowledge you might have! Make your responses as concise as possible."
        "Question: Viscount Yamaji Motoharu was a general in the early Imperial Japanese Army which belonged to which Empire?\n"
        "Knowledge Triplets: Imperial Japanese Army, allegiance, Emperor of Japan\n"
        "Yamaji Motoharu, allegiance, Emperor of Japan\n"
        "Yamaji Motoharu, military rank, general\n"
        "Answer: {Yes}. The answer is {Empire of Japan}.\n\n"
        "Question: Which psychedelic compound found in the Psilocybin genus mushroom is converted to psilocin by the body?\n"
        "Knowledge Triplets: psilocybin, described by source, Opium Law\n"
        "psilocybin, found in taxon, Gymnopilus purpuratus\n"
        "psilocybin, found in taxon, Gymnopilus spectabilis\n"
        "Opium Law, part of, norcodeine (stereochemistry defined)\n"
        "Gymnopilus purpuratus, edibility, psychoactive mushroom\n"
        "Gymnopilus spectabilis, parent taxon, Gymnopilus\n"
        "Answer: {No}.\n\n"
        "Question: Which tennis player is younger, John Newcombe or Květa Peschke?\n"
        "Knowledge Triplets: Květa Peschke, date of birth, +1975-07-09T00:00:00Z\n"
        "John Newcombe, date of birth, +1944-05-23T00:00:00Z\n"
        "John Newcombe, country of citizenship, Australia\n"
        "Answer: {Yes}. The answer is {Květa Peschke}.\n\n"
        "Question: At what stadium did Mychal George Thompson play home games with the San Antonio Spurs?\n"
        "Knowledge Triplets: San Antonio Spurs, home venue, AT&T Center\n"
        "San Antonio Spurs, home venue, Alamodome\n"
        "San Antonio Spurs, home venue, Fort Worth Convention Center\n"
        "AT&T Center, occupant, San Antonio Spurs\n"
        "Fort Worth Convention Center, located in the administrative territorial entity, Texas\n"
        "Fort Worth Convention Center, occupant, San Antonio Spurs\n"
        "Answer: {Yes}. The answer is {AT&T Center}.\n\n"
    )

def rog_style_answer_prompt():
    return ("Based on the provided evidence, in the form of knowledge graph triples (subject, relation, object), please answer the question. Please keep the answer as simple as possible and return a single answer only.\n")


def answer_prompt():
    return (
        "Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to answer the question by combining the information contained in the triplets with your own knowledge.\n"
        "Question: Viscount Yamaji Motoharu was a general in the early Imperial Japanese Army which belonged to which Empire?\n"
        "Knowledge Triplets: Imperial Japanese Army, allegiance, Emperor of Japan\n"
        "Yamaji Motoharu, allegiance, Emperor of Japan\n"
        "Yamaji Motoharu, military rank, general\n"
        "Answer: Based on the given knowledge triplets and my knowledge, Viscount Yamaji Motoharu, who was a general in the early Imperial Japanese Army, belonged to the Empire of Japan. Therefore, the answer to the question is {Empire of Japan}.\n\n"
        "Question: Which psychedelic compound found in the Psilocybin genus mushroom is converted to psilocin by the body?\n"
        "Knowledge Triplets: psilocybin, described by source, Opium Law\n"
        "psilocybin, found in taxon, Gymnopilus purpuratus\n"
        "psilocybin, found in taxon, Gymnopilus spectabilis\n"
        "Opium Law, part of, norcodeine (stereochemistry defined)\n"
        "Gymnopilus purpuratus, edibility, psychoactive mushroom\n"
        "Gymnopilus spectabilis, parent taxon, Gymnopilus\n"
        "Answer: Based on the given knowledge triplets and my knowledge, the specific psychedelic compound found in the Psilocybin genus mushroom that is converted to psilocin by the body is not explicitly mentioned. Therefore, additional knowledge about the specific compounds and their conversion to psilocin is required to answer the question.\n\n"
        "Question: Which tennis player is younger, John Newcombe or Květa Peschke?\n"
        "Knowledge Triplets: Květa Peschke, date of birth, +1975-07-09T00:00:00Z\n"
        "John Newcombe, date of birth, +1944-05-23T00:00:00Z\n"
        "John Newcombe, country of citizenship, Australia\n"
        "Answer: Based on the given knowledge triplets and my knowledge, John Newcombe was born on May 23, 1944, and Květa Peschke was born on July 9, 1975. Therefore, {Květa Peschke} is younger than John Newcombe.\n\n"
        "Question: At what stadium did Mychal George Thompson play home games with the San Antonio Spurs?\n"
        "Knowledge Triplets: San Antonio Spurs, home venue, AT&T Center\n"
        "San Antonio Spurs, home venue, Alamodome\n"
        "San Antonio Spurs, home venue, Fort Worth Convention Center\n"
        "AT&T Center, occupant, San Antonio Spurs\n"
        "Fort Worth Convention Center, located in the administrative territorial entity, Texas\n"
        "Fort Worth Convention Center, occupant, San Antonio Spurs\n"
        "Answer: Based on the given knowledge triplets and my knowledge, Mychal George Thompson played home games with the San Antonio Spurs at the AT&T Center. Therefore, the answer to the question is {AT&T Center}.\n\n"
    )



def cot_prompt(question):
    return [
        {
            "role": "user",
            "content": "Question: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?",
        },
        {
            "role": "assistant",
            "content": "Answer: First, the education institution has a sports team named George Washington Colonials men's basketball in is George Washington University. Second, George Washington University is in Washington D.C. The answer is {Washington, D.C.}",
        },
        {
            "role": "user",
            "content": "Question: Who lists Pramatha Chaudhuri as an influence and wrote Jana Gana Mana?",
        },
        {
            "role": "assistant",
            "content": "Answer: First, Bharoto Bhagyo Bidhata wrote Jana Gana Mana. Second, Bharoto Bhagyo Bidhata lists Pramatha Chaudhuri as an influence. The answer is {Bharoto Bhagyo Bidhata}.",
        },
        {
            "role": "user",
            "content": "Question: Who was the artist nominated for an award for You Drive Me Crazy?",
        },
        {
            "role": "assistant",
            "content": "Answer: First, the artist nominated for an award for You Drive Me Crazy is Britney Spears. The answer is {Britney Spears}.",
        },
        {
            "role": "user",
            "content": "Question: What person born in Siegen influenced the work of Vincent Van Gogh?",
        },
        {
            "role": "assistant",
            "content": "Answer: First, Peter Paul Rubens, Claude Monet and etc. influenced the work of Vincent Van Gogh. Second, Peter Paul Rubens born in Siegen. The answer is {Peter Paul Rubens}.",
        },
        {"role": "user", "content": f"Question: {question}"},
    ]


def io_prompt(question):
    return [
        {
            "role": "user",
            "content": "Question: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?",
        },
        {
            "role": "assistant",
            "content": "Answer: {Washington, D.C.}.",
        },
        {
            "role": "user",
            "content": "Question: Who lists Pramatha Chaudhuri as an influence and wrote Jana Gana Mana?",
        },
        {
            "role": "assistant",
            "content": "Answer: {Bharoto Bhagyo Bidhata}.",
        },
        {
            "role": "user",
            "content": "Question: Who was the artist nominated for an award for You Drive Me Crazy?",
        },
        {
            "role": "assistant",
            "content": "Answer: {Britney Spears}.",
        },
        {
            "role": "user",
            "content": "Question: What person born in Siegen influenced the work of Vincent Van Gogh?",
        },
        {
            "role": "assistant",
            "content": "Answer: {Peter Paul Rubens}.",
        },
        {"role": "user", "content": f"Question: {question}"},
    ]


def tog_answer_prompt(query: str, subgraph, kg) -> str:
    prompt = rog_style_answer_prompt()
    prompt += (
        f"Question: {query}\n"
        f"Knowledge Triplets: "
    )
    for h, r, t in subgraph:
        prompt += (
            f"{kg.get_node_label(h)}, "
            f"{kg.get_relation_label(r)}, "
            f"{kg.get_node_label(t)}\n"
        )
    prompt += "Answer: "
    return prompt
