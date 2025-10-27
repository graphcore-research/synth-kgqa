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

# Partially adapted from the prompts in https://github.com/GasolSun36/ToG/blob/main/ToG/prompt_list.py


def n_hop_prompt(n_hops, flattened_graph):
    assert n_hops < 5, "Only prompts for up to four edges implemented"
    return {
        1: one_hop_prompt,
        2: two_hop_prompt,
        3: three_hop_prompt,
        4: four_hop_prompt,
    }[n_hops](flattened_graph)


def one_hop_prompt(flattened_graph):
    return [
        {
            "role": "user",
            "content": "Based on the provided set of knowledge graph triples, please generate a question that can be answered using exactly one of the provided triples. The answer should correspond to exactly one node in the provided graph, be unique and not ambiguous. Apply your internal knowledge and common sense to ensure that you respond with a question that has a unique answer, not only within the provided graph but overall. i.e. if there the question has another correct answer the question is invalid, even if the other answer is not present in the knowledge graph. Respond only with "
            "'Question:' the generated question,\n"
            "'Nodes mentioned in the question:' a semicolon-separated list of the nodes that are explicitly mentioned in the question,\n"
            "'Answer:' The node corresponding to the correct answer with its QID,\n"
            "'Triples used:' a semicolon-separated list of the triples used for answering the question,\n"
            "'SPARQL query:' a SPARQL query to return all answers of the question from the WikiData knowledge base",
        },
        {
            "role": "user",
            "content": "Graph: Ludwig van Beethoven (Q12368917)-child (P40)-Johann van Beethoven (Q2153541);Johann van Beethoven (Q2153541)-father (P22)-Ludwig van Beethoven (Q12368917);Ludwig van Beethoven (Q12368917)-country of origin (P495)-Netherlands (Q55);Ludwig van Beethoven (Q12368917)-country of citizenship (P27)-Netherlands (Q55);Johann van Beethoven (Q2153541)-native language (P103)-German (Q188);Johann van Beethoven (Q2153541)-languages spoken, written or signed (P1412)-German (Q188);Johann Gottlob Benjamin Pfeil (Q1256273)-languages spoken, written or signed (P1412)-German (Q188);Ludwig van Beethoven (Q12368917)-occupation (P106)-musician (Q639669);Johann van Beethoven (Q2153541)-occupation (P106)-musician (Q639669);Johann van Beethoven (Q2153541)-place of birth (P19)-Bonn (Q586);Johann van Beethoven (Q2153541)-work location (P937)-Bonn (Q586);Johann van Beethoven (Q2153541)-place of death (P20)-Bonn (Q586);Ludwig van Beethoven (Q12368917)-place of death (P20)-Bonn (Q586)",
        },
        {
            "role": "assistant",
            "content": "Question: Who was the father of Ludwig van Beethoven?\n"
            "Nodes mentioned in the question: Ludwig van Beethoven (Q12368917)\n"
            "Answer: Johann van Beethoven (Q2153541)\n"
            "Triples used: Johann van Beethoven (Q2153541)-father (P22)-Ludwig van Beethoven (Q12368917)\n"
            "SPARQL query: SELECT ?answer WHERE {?answer wdt:P22 wd:Q12368917.}",
        },
        {
            "role": "user",
            "content": "Graph: Hugoton (Q1004784)-country (P17)-United States of America (Q30);Hugoton (Q1004784)-located in the administrative territorial entity (P131)-Stevens County (Q402259);Stevens County (Q402259)-country (P17)-United States of America (Q30);Samuel Newitt Wood (Q7412275)-country of citizenship (P27)-United States of America (Q30);Samuel Newitt Wood (Q7412275)-place of death (P20)-Hugoton (Q1004784);New Jersey Route 171 (Q827205)-country (P17)-United States of America (Q30);Louise Allbritton (Q15060610)-country of citizenship (P27)-United States of America (Q30);Samuel Newitt Wood (Q7412275)-member of political party (P102)-Republican Party (Q29468);Republican Party (Q29468)-country (P17)-United States of America (Q30)",
        },
        {
            "role": "assistant",
            "content": "Question: What was Samuel Newitt Wood's political party?\n"
            "Nodes mentioned in the question: Samuel Newitt Wood (Q7412275)\n"
            "Answer: Republican Party (Q29468)\n"
            "Triples used: Samuel Newitt Wood (Q7412275)-member of political party (P102)-Republican Party (Q29468)\n"
            "SPARQL query: SELECT ?answer WHERE {wd:Q7412275 wdt:P102 ?answer.}",
        },
        {
            "role": "user",
            "content": "Graph: William Brown (Q8006010)-place of birth (P19)-Melton Mowbray (Q1018423);Claude Lorraine Ferneley (Q19833247)-place of death (P20)-Melton Mowbray (Q1018423);Claude Lorraine Ferneley (Q19833247)-place of birth (P19)-Melton Mowbray (Q1018423);William Brown (Q8006010)-country of citizenship (P27)-United Kingdom (Q145);Claude Lorraine Ferneley (Q19833247)-country of citizenship (P27)-United Kingdom (Q145);Melton Mowbray (Q1018423)-country (P17)-United Kingdom (Q145);Claude Lorraine Ferneley (Q19833247)-named after (P138)-Claude Lorrain (Q214074);Frederick Selous (Q1306437)-country of citizenship (P27)-United Kingdom (Q145);William Brown (Q8006010)-occupation (P106)-military personnel (Q47064)",
        },
        {
            "role": "assistant",
            "content": "Question: Where did Claude Lorraine Ferneley die?\n"
            "Nodes mentioned in the question: Claude Lorraine Ferneley (Q19833247)\n"
            "Answer: Melton Mowbray (Q1018423)\n"
            "Triples used: Claude Lorraine Ferneley (Q19833247)-place of death (P20)-Melton Mowbray (Q1018423)\n"
            "SPARQL query: SELECT ?answer WHERE {wd:Q19833247 wdt:P20 ?answer.}",
        },
        {
            "role": "user",
            "content": "Graph: Virgin of the Rocks (Q11935346)-creator (P170)-Leonardo da Vinci (Q762);Virgin of the Rocks (Q11935346)-genre (P136)-religious painting (Q2414609);Lamentation of Christ (Q19291834)-genre (P136)-religious painting (Q2414609);The Virgin and Child with Saint Anne (Q563727)-genre (P136)-religious painting (Q2414609);Leonardo da Vinci (Q762)-notable work (P800)-The Virgin and Child with Saint Anne (Q563727);Paris Manuscript B (Q11767217)-author (P50)-Leonardo da Vinci (Q762);The Virgin and Child with Saint Anne and Saint John the Baptist (Q2471531)-genre (P136)-religious painting (Q2414609);The Virgin and Child with Saint Anne and Saint John the Baptist (Q2471531)-creator (P170)-Leonardo da Vinci (Q762)",
        },
        {
            "role": "assistant",
            "content": "Question: Who is the author of The Virgin and Child with Saint Anne?\n"
            "Nodes mentioned in the question: The Virgin and Child with Saint Anne (Q563727)\n"
            "Answer: Leonardo da Vinci (Q762)\n"
            "Triples used: Leonardo da Vinci (Q762)-notable work (P800)-The Virgin and Child with Saint Anne (Q563727)\n"
            "SPARQL query: SELECT ?answer WHERE {?answer wdt:P800 wd:Q563727.}",
        },
        {"role": "user", "content": f"Graph: {flattened_graph}"},
    ]


def two_hop_prompt(flattened_graph):
    return [
        {
            "role": "user",
            "content": "Based on the provided set of knowledge graph triples, please generate a question that requires combining the information in exactly two of the provided triples for answering. The answer should correspond to exactly one node in the provided graph, be unique and not ambiguous. Make sure that both triples are required for answering the question and that the two triples involve three different entities. Respond only with "
            "'Question:' the generated question,\n"
            "'Nodes mentioned in the question:' a semicolon-separated list of the nodes that are explicitly mentioned in the question,\n"
            "'Answer:' The node corresponding to the correct answer with its QID,\n"
            "'Triples used:' a semicolon-separated list of the triples used for answering the question,\n"
            "'SPARQL query:' a SPARQL query to return all answers of the question from the WikiData knowledge base",
        },
        {
            "role": "user",
            "content": "Graph: Quiaios (Q1024001)-country (P17)-Portugal (Q45);Savage Nights (Q174785)-filming location (P915)-Portugal (Q45);Quiaios (Q1024001)-located in the administrative territorial entity (P131)-Figueira da Foz (Q428459);Figueira da Foz (Q428459)-country (P17)-Portugal (Q45);Savage Nights (Q174785)-filming location (P915)-Lisbon (Q597);Portugal (Q45)-capital (P36)-Lisbon (Q597);Lisbon (Q597)-country (P17)-Portugal (Q45);Lisbon (Q597)-capital of (P1376)-Portugal (Q45);Francisco José Fernandes Costa (Q357932)-country of citizenship (P27)-Portugal (Q45);Francisco José Fernandes Costa (Q357932)-place of death (P20)-Figueira da Foz (Q428459);Francisco José Fernandes Costa (Q357932)-position held (P39)-Minister of Foreign Affairs (Q4294919);Minister of Foreign Affairs (Q4294919)-country (P17)-Portugal (Q45);Minister of Foreign Affairs (Q4294919)-applies to jurisdiction (P1001)-Portugal (Q45);Francisco José Fernandes Costa (Q357932)-occupation (P106)-politician (Q82955)",
        },
        {
            "role": "assistant",  # Iso-1
            "content": "Question: In which country did Francisco José Fernandes Costa die?\n"
            "Nodes mentioned in the question: Francisco José Fernandes Costa (Q357932)\n"
            "Answer: Portugal (Q45)\n"
            "Triples used: Francisco José Fernandes Costa (Q357932)-place of death (P20)-Figueira da Foz (Q428459); Figueira da Foz (Q428459)-country (P17)-Portugal (Q45)\n"
            "SPARQL query: SELECT ?answer WHERE {wd:Q357932 wdt:P20 ?place. ?place wdt:P17 ?answer.}",
        },
        {
            "role": "user",
            "content": "Graph: Frederic M. Scherer (Q5497093)-languages spoken, written or signed-English (Q1860);McLaren's Negatives (Q6802050)-original language of film or TV show-English (Q1860);Over the Moon (Q7113500)-original language of film or TV show-English (Q1860);Antony Kamm (Q2907643)-languages spoken, written or signed-English (Q1860);Over the Moon (Q7113500)-cast member-Merle Oberon (Q230958);McLaren's Negatives (Q6802050)-director of photography-Marie-Josée Saint-Pierre (Q6762681);McLaren's Negatives (Q6802050)-director-Marie-Josée Saint-Pierre (Q6762681);McLaren's Negatives (Q6802050)-producer-Marie-Josée Saint-Pierre (Q6762681);McLaren's Negatives (Q6802050)-genre-documentary film (Q93204);The Mystery Squadron (Q7752881)-original language of film or TV show-English (Q1860);Over the Moon (Q7113500)-screenwriter-Robert E. Sherwood (Q1362097);Marie-Josée Saint-Pierre (Q6762681)-country of citizenship-Canada (Q16);McLaren's Negatives (Q6802050)-country of origin-Canada (Q16);Canada (Q16)-official language-English (Q1860);Carol Sealey (Q3660517)-country of citizenship-Canada (Q16);Désirée (Q945470)-cast member-Alan Napier (Q373306);Zaritsas: Russian Women in New York (Q8066803)-genre-documentary film (Q93204);Zaritsas: Russian Women in New York (Q8066803)-original language of film or TV show-English (Q1860);Marie-Josée Saint-Pierre (Q6762681)-field of work-film producer (Q3282637);Frederic M. Scherer (Q5497093)-educated at-Harvard University (Q13371);Frederic M. Scherer (Q5497093)-employer-Harvard University (Q13371);Robert E. Sherwood (Q1362097)-educated at-Harvard University (Q13371);Buster Keaton Rides Again (Q2928788)-country of origin-Canada (Q16);Buster Keaton Rides Again (Q2928788)-genre-documentary film (Q93204);Buster Keaton Rides Again (Q2928788)-original language of film or TV show-English (Q1860);Buster Keaton Rides Again (Q2928788)-main subject-cinema of Canada (Q2973157);Buster Keaton Rides Again (Q2928788)-main subject-actor (Q33999)",
        },
        {
            "role": "assistant",  # Iso-1
            "content": "Question: Where was the screenwriter of Over the Moon educated?\n"
            "Nodes mentioned in the question: Over the Moon (Q7113500)\n"
            "Answer: Harvard University (Q13371)\n"
            "Triples used: Over the Moon (Q7113500)-screenwriter-Robert E. Sherwood (Q1362097); Robert E. Sherwood (Q1362097)-educated at-Harvard University (Q13371)\n"
            "SPARQL query: SELECT ?answer WHERE {wd:Q7113500 wdt:P58 ?person. ?person wdt:P69 ?answer.}",
        },
        {
            "role": "user",
            "content": "Graph: Tokmak Raion (Q2216309)-country (P17)-Ukraine (Q212);Luhan River (Q2639341)-country (P17)-Ukraine (Q212);Battalions of territorial defense in Ukraine (Q17622327)-country (P17)-Ukraine (Q212);Luhansk (Q134279)-country (P17)-Ukraine (Q212);Luhansk (Q134279)-named after (P138)-Luhan River (Q2639341);Zaporizhzhia Oblast (Q171334)-country (P17)-Ukraine (Q212);Zaporizhzhia Oblast (Q171334)-located in the administrative territorial entity (P131)-Ukraine (Q212);Ukraine (Q212)-contains administrative territorial entity (P150)-Zaporizhzhia Oblast (Q171334);Tokmak Raion (Q2216309)-located in the administrative territorial entity (P131)-Zaporizhzhia Oblast (Q171334);Alexander Molodchy (Q1981239)-country of citizenship (P27)-Ukraine (Q212);Alexander Molodchy (Q1981239)-place of birth (P19)-Luhansk (Q134279);Donets River (Q191238)-country (P17)-Ukraine (Q212);Luhan River (Q2639341)-mouth of the watercourse (P403)-Donets River (Q191238);Donets River (Q191238)-tributary (P974)-Luhan River (Q2639341);Starobilsk (Q987345)-country (P17)-Ukraine (Q212);Battalions of territorial defense in Ukraine (Q17622327)-subclass of (P279)-battalion (Q6382533);Vilniansk Raion (Q2216231)-country (P17)-Ukraine (Q212);Vilniansk Raion (Q2216231)-located in the administrative territorial entity (P131)-Zaporizhzhia Oblast (Q171334);company (Q102538)-part of (P361)-battalion (Q6382533);Black Sea (Q166)-shares border with (P47)-Zaporizhzhia Oblast (Q171334);Zaporizhzhia Oblast (Q171334)-shares border with (P47)-Black Sea (Q166);Black Sea (Q166)-country (P17)-Ukraine (Q212);battalion (Q6382533)-subclass of (P279)-military unit (Q176799);company (Q102538)-subclass of (P279)-military unit (Q176799)",
        },
        {
            "role": "assistant",  # Iso-1
            "content": "Question: What was the place of birth of Alexander Molodchy named after?\n"
            "Nodes mentioned in the question: Alexander Molodchy (Q1981239)\n"
            "Answer: Luhan River (Q2639341)\n"
            "Triples used: Alexander Molodchy (Q1981239)-place of birth (P19)-Luhansk (Q134279); Luhansk (Q134279)-named after (P138)-Luhan River (Q2639341)\n"
            "SPARQL query: SELECT ?answer WHERE {wd:Q1981239 wdt:P19 ?place. ?place wdt:P138 ?answer.}",
        },
        {
            "role": "user",
            "content": "Graph: Wataru Tagami (Q11575134)-member of sports team (P54)-V-Varen Nagasaki (Q3029309);Wataru Tagami (Q11575134)-occupation (P106)-association football player (Q937857);Wataru Tagami (Q11575134)-place of birth (P19)-Kitakyūshū (Q188806);Kokura Station (Q801027)-located in the administrative territorial entity (P131)-Kitakyūshū (Q188806);Eric Bhamuza Sono (Q5386133)-occupation (P106)-association football player (Q937857);Nagasaki Athletic Stadium (Q6958603)-occupant (P466)-V-Varen Nagasaki (Q3029309);V-Varen Nagasaki (Q3029309)-home venue (P115)-Nagasaki Athletic Stadium (Q6958603);Kitakyūshū (Q188806)-country (P17)-Japan (Q17);Nagasaki Athletic Stadium (Q6958603)-country (P17)-Japan (Q17);Kokura Station (Q801027)-country (P17)-Japan (Q17);Wataru Tagami (Q11575134)-country of citizenship (P27)-Japan (Q17)",
        },
        {
            "role": "assistant",  # Iso-2
            "content": "Question: Which member of V-Varen Nagasaki was born in Kitakyūshū?\n"
            "Nodes mentioned in the question: V-Varen Nagasaki (Q3029309); Kitakyūshū (Q188806)\n"
            "Answer: Wataru Tagami (Q11575134)\n"
            "Triples used: Wataru Tagami (Q11575134)-member of sports team (P54)-V-Varen Nagasaki (Q3029309); Wataru Tagami (Q11575134)-place of birth (P19)-Kitakyūshū (Q188806)\n"
            "SPARQL query: SELECT ?answer WHERE {?answer wdt:P54 wd:Q3029309. ?answer wdt:P19 wd:Q188806.}",
        },
        {
            "role": "user",
            "content": "Graph: To the Nameless Dead (Q1767972)-follows (P155)-The Gathering Wilderness (Q1002176);The Gathering Wilderness (Q1002176)-followed by (P156)-To the Nameless Dead (Q1767972);To the Nameless Dead (Q1767972)-performer (P175)-Primordial (Q1345756);The Gathering Wilderness (Q1002176)-performer (P175)-Primordial (Q1345756);Redemption at the Puritan's Hand (Q3702054)-follows (P155)-To the Nameless Dead (Q1767972);To the Nameless Dead (Q1767972)-followed by (P156)-Redemption at the Puritan's Hand (Q3702054);Redemption at the Puritan's Hand (Q3702054)-performer (P175)-Primordial (Q1345756);Where Greater Men Have Fallen (Q19248440)-follows (P155)-Redemption at the Puritan's Hand (Q3702054);Where Greater Men Have Fallen (Q19248440)-performer (P175)-Primordial (Q1345756);Where Greater Men Have Fallen (Q19248440)-genre (P136)-doom metal (Q186170);Assembly (Q1950989)-genre (P136)-doom metal (Q186170);Redemption at the Puritan's Hand (Q3702054)-record label (P264)-Metal Blade Records (Q756320);To the Nameless Dead (Q1767972)-record label (P264)-Metal Blade Records (Q756320);The Gathering Wilderness (Q1002176)-record label (P264)-Metal Blade Records (Q756320);Where Greater Men Have Fallen (Q19248440)-record label (P264)-Metal Blade Records (Q756320);Storm Before Calm (Q1767899)-followed by (P156)-The Gathering Wilderness (Q1002176);Storm Before Calm (Q1767899)-performer (P175)-Primordial (Q1345756);The Walking Dead (Q7773282)-genre (P136)-doom metal (Q186170)",
        },
        {
            "role": "assistant",  # Iso-2
            "content": "Question: Which album performed by Primordial follows Storm Before Calm?\n"
            "Nodes mentioned in the question: Primordial (Q1345756); Storm Before Calm (Q1767899)\n"
            "Answer: The Gathering Wilderness (Q1002176)\n"
            "Triples used: Storm Before Calm (Q1767899)-followed by (P156)-The Gathering Wilderness (Q1002176); The Gathering Wilderness (Q1002176)-performer (P175)-Primordial (Q1345756)\n"
            "SPARQL query: SELECT ?answer WHERE {wd:Q1767899 wdt:P156 ?answer. ?answer wdt:P175 wd:Q1345756.}",
        },
        {"role": "user", "content": f"Graph: {flattened_graph}"},
    ]


def three_hop_prompt(flattened_graph):
    return [
        {
            "role": "user",
            "content": "Based on the provided set of knowledge graph triples, please generate a question that requires combining the information in exactly three of the provided triples for answering. The answer should correspond to exactly one node in the provided graph, be unique and not ambiguous. Make sure that all three selected triples are required for answering the question and that the three triples involve at least four different entities. Make sure also that there isn't a smallest set of triples in the graph which is enough to deduce the answer. Respond only with "
            "'Question:' the generated question,\n"
            "'Nodes mentioned in the question:' a semicolon-separated list of the nodes that are explicitly mentioned in the question,\n"
            "'Answer:' The node corresponding to the correct answer with its QID,\n"
            "'Triples used:' a semicolon-separated list of the triples used for answering the question,\n"
            "'SPARQL query:' a SPARQL query to return all answers of the question from the WikiData knowledge base",
        },
        {
            "role": "user",
            "content": "Graph: Leonardo da Vinci (Q762)-occupation (P106)-philosopher (Q4964182);The Battle of Anghiari (Q2045726)-creator (P170)-Leonardo da Vinci (Q762);Friedrich Siegmund-Schultze (Q105593)-occupation (P106)-philosopher (Q4964182);Paris Manuscript K (Q11767225)-author (P50)-Leonardo da Vinci (Q762);The Tiger Hunt (Q544488)-inspired by (P941)-The Battle of Anghiari (Q2045726);Paris Manuscripts (Q1784170)-author (P50)-Leonardo da Vinci (Q762);Paris Manuscripts (Q1784170)-has part (P527)-Paris Manuscript K (Q11767225);Paris Manuscript K (Q11767225)-part of (P361)-Paris Manuscripts (Q1784170);Richard Shore (Q763124)-occupation (P106)-philosopher (Q4964182);The Battle of Anghiari (Q2045726)-country (P17)-Italy (Q38);Pseudo-Dionysius the Areopagite (Q312816)-occupation (P106)-philosopher (Q4964182);Paris Manuscripts (Q1784170)-collection (P195)-Institut de France (Q377066);Paris Manuscript K (Q11767225)-collection (P195)-Institut de France (Q377066);The Battle of Anghiari (Q2045726)-movement (P135)-High Renaissance (Q1474884);Leonardo da Vinci (Q762)-movement (P135)-High Renaissance (Q1474884)",
        },
        {
            "role": "assistant",
            "content": "Question: Which artist from the High Renaissance movement is the author of work included in the collection of the Institut de France?\n"
            "Nodes mentioned in the question: High Renaissance (Q1474884); Institut de France (Q377066)\n"
            "Answer: Leonardo da Vinci (Q762)\n"
            "Triples used: Leonardo da Vinci (Q762)-movement (P135)-High Renaissance (Q1474884); Paris Manuscripts (Q1784170)-author (P50)-Leonardo da Vinci (Q762); Paris Manuscripts (Q1784170)-collection (P195)-Institut de France (Q377066)\n"
            "SPARQL query: SELECT ?answer WHERE {?answer wdt:P135 wd:Q1474884. ?work wdt:P195 wd:Q377066. ?work wdt:P50 ?answer.}",
        },
        {
            "role": "user",
            "content": "Graph: Peter Kramer (Q100212)-occupation (P106)-physicist (Q169470);Peter Kramer (Q100212)-educated at (P69)-University of Tübingen (Q153978);Peter Kramer (Q100212)-employer (P108)-University of Tübingen (Q153978);Peter Kramer (Q100212)-place of birth (P19)-Quedlinburg (Q40623);Christoph Vitzthum von Eckstädt (Q1086043)-place of birth (P19)-Quedlinburg (Q40623);Christoph Vitzthum von Eckstädt (Q1086043)-occupation (P106)-chamberlain (Q264323);Kurt Herzog (Q3200585)-place of birth (P19)-Quedlinburg (Q40623);Kurt Herzog (Q3200585)-country of citizenship (P27)-Germany (Q183);University of Tübingen (Q153978)-country (P17)-Germany (Q183);Christoph Vitzthum von Eckstädt (Q1086043)-country of citizenship (P27)-Germany (Q183);Quedlinburg (Q40623)-country (P17)-Germany (Q183);Peter Kramer (Q100212)-country of citizenship (P27)-Germany (Q183);Schönwölkau (Q16065)-country (P17)-Germany (Q183);Christoph Vitzthum von Eckstädt (Q1086043)-place of death (P20)-Schönwölkau (Q16065);Peter Kramer (Q100212)-educated at (P69)-University of Bristol (Q459506);Kurt Herzog (Q3200585)-place of death (P20)-Vorkuta (Q133529);Quedlinburg (Q40623)-located in the administrative territorial entity (P131)-Harz District (Q6087);Harz District (Q6087)-country (P17)-Germany (Q183)",
        },
        {
            "role": "assistant",
            "content": "Question: Who is the chamberlain that was born in the Harz District?\n"
            "Nodes mentioned in the question: chamberlain (Q264323); Harz District (Q6087)\n"
            "Answer: Christoph Vitzthum von Eckstädt (Q1086043)\n"
            "Triples used: Christoph Vitzthum von Eckstädt (Q1086043)-occupation (P106)-chamberlain (Q264323); Christoph Vitzthum von Eckstädt (Q1086043)-place of birth (P19)-Quedlinburg (Q40623); Quedlinburg (Q40623)-located in the administrative territorial entity (P131)-Harz District (Q6087)\n"
            "SPARQL query: SELECT ?answer WHERE {?answer wdt:P19 ?place. ?place wdt:P131 wd:Q6087. ?answer wdt:P106 wd:Q264323}",
        },
        {
            "role": "user",
            "content": "Graph: Virginia D. Smith (Q113955)-place of death (P20)-Sun City West (Q1024100);Virginia D. Smith (Q113955)-country of citizenship (P27)-United States of America (Q30);Sun City West (Q1024100)-country (P17)-United States of America (Q30);Virginia D. Smith (Q113955)-family name (P734)-Smith (Q1158446);Aquatic Elephants (Q17520567)-country of origin (P495)-United States of America (Q30);Virginia D. Smith (Q113955)-position held (P39)-United States representative (Q13218630);United States representative (Q13218630)-country (P17)-United States of America (Q30);United States representative (Q13218630)-applies to jurisdiction (P1001)-United States of America (Q30);Dick Holub (Q3706951)-place of death (P20)-Sun City West (Q1024100);Dick Holub (Q3706951)-country of citizenship (P27)-United States of America (Q30);Aquatic Elephants (Q17520567)-genre (P136)-documentary film (Q93204);Virginia D. Smith (Q113955)-occupation (P106)-politician (Q82955);Virginia D. Smith (Q113955)-member of political party (P102)-Republican Party (Q29468);Republican Party (Q29468)-country (P17)-United States of America (Q30);Sigrid Schneider (Q2285318)-occupation (P106)-politician (Q82955);Jean-Pierre Smith (Q16233581)-family name (P734)-Smith (Q1158446)",
        },
        {
            "role": "assistant",
            "content": "Question: What was the political party of membership of the United States representative who died in Sun City West?\n"
            "Nodes mentioned in the question: United States representative (Q13218630); Sun City West (Q1024100)\n"
            "Answer: Republican Party (Q29468)\n"
            "Triples used: Virginia D. Smith (Q113955)-place of death (P20)-Sun City West (Q1024100); Virginia D. Smith (Q113955)-position held (P39)-United States representative (Q13218630); Virginia D. Smith (Q113955)-member of political party (P102)-Republican Party (Q29468)\n"
            "SPARQL query: SELECT ?answer WHERE {?person wdt:P20 wd:Q1024100. ?person wdt:P39 wd:Q13218630. ?person wdt:P102 ?answer}",
        },
        {
            "role": "user",
            "content": "Graph: Ossos (Q10342302)-original language of film or TV show (P364)-Portuguese (Q5146);Modern Life (Q3213681)-producer (P162)-Paulo Branco (Q259593);Mário Cravo Neto (Q2373371)-languages spoken, written or signed (P1412)-Portuguese (Q5146);Brazil (Q155)-official language (P37)-Portuguese (Q5146);Ossos (Q10342302)-producer (P162)-Paulo Branco (Q259593);Mário Cravo Neto (Q2373371)-country of citizenship (P27)-Brazil (Q155);Nelson Piquet (Q171346)-country of citizenship (P27)-Brazil (Q155);Salvador (Q36947)-country (P17)-Brazil (Q155);Mário Cravo Neto (Q2373371)-place of birth (P19)-Salvador (Q36947);Ossos (Q10342302)-genre (P136)-art film (Q1135802);Nelson Piquet (Q171346)-occupation (P106)-racing driver (Q378622);Tony Dron (Q7822260)-occupation (P106)-racing driver (Q378622);Ossos (Q10342302)-director (P57)-Pedro Costa (Q1354061);Pedro Costa (Q1354061)-languages spoken, written or signed (P1412)-Portuguese (Q5146)",
        },
        {
            "role": "assistant",
            "content": "Question: What is the genre of the movie produced by Paulo Branco and directed by Pedro Costa?\n"
            "Nodes mentioned in the question: Paulo Branco (Q259593); Pedro Costa (Q1354061)\n"
            "Answer: art film (Q1135802)\n"
            "Triples used: Ossos (Q10342302)-producer (P162)-Paulo Branco (Q259593); Ossos (Q10342302)-director (P57)-Pedro Costa (Q1354061); Ossos (Q10342302)-genre (P136)-art film (Q1135802)\n"
            "SPARQL query: SELECT ?answer WHERE {?movie wdt:P162 wd:Q259593. ?movie wdt:P57 wd:Q1354061. ?movie wdt:P136 ?answer}",
        },
        {
            "role": "user",
            "content": "Graph: Dan Brown (Q7345)-educated at (P69)-Amherst College (Q49165);Be Cool (Q572903)-cast member (P161)-Dan Brown (Q7345);Dan Brown (Q7345)-notable work (P800)-The Lost Symbol (Q211006);The Lost Symbol (Q211006)-author (P50)-Dan Brown (Q7345);Alexander Brough (Q15996873)-educated at (P69)-Amherst College (Q49165);Dan Brown (Q7345)-notable work (P800)-Inferno (Q4845793);Inferno (Q4845793)-author (P50)-Dan Brown (Q7345);The Lost Symbol (Q211006)-followed by (P156)-Inferno (Q4845793);Inferno (Q4845793)-follows (P155)-The Lost Symbol (Q211006);Harold E. Varmus (Q295691)-educated at (P69)-Amherst College (Q49165);Inferno (Q4845793)-characters (P674)-Robert Langdon (Q210548);Inferno (Q4845793)-part of the series (P179)-Robert Langdon (Q210548);The Lost Symbol (Q211006)-part of the series (P179)-Robert Langdon (Q210548);The Lost Symbol (Q211006)-characters (P674)-Robert Langdon (Q210548);The Da Vinci Code (Q81689)-followed by (P156)-The Lost Symbol (Q211006);The Lost Symbol (Q211006)-follows (P155)-The Da Vinci Code (Q81689);The Da Vinci Code (Q81689)-part of the series (P179)-Robert Langdon (Q210548);The Da Vinci Code (Q81689)-characters (P674)-Robert Langdon (Q210548);The Da Vinci Code (Q81689)-author (P50)-Dan Brown (Q7345);Dan Brown (Q7345)-notable work (P800)-The Da Vinci Code (Q81689);Amherst College (Q49165)-country (P17)-United States of America (Q30);Inferno (Q4845793)-original language of film or TV show (P364)-English (Q1860);The Da Vinci Code (Q81689)-original language of film or TV show (P364)-English (Q1860);Be Cool (Q572903)-original language of film or TV show (P364)-English (Q1860);The Lost Symbol (Q211006)-original language of film or TV show (P364)-English (Q1860);Dan Brown (Q7345)-native language (P103)-English (Q1860);Dan Brown (Q7345)-languages spoken, written or signed (P1412)-English (Q1860);Harold E. Varmus (Q295691)-languages spoken, written or signed (P1412)-English (Q1860);John Coolidge (Q6227035)-educated at (P69)-Amherst College (Q49165);Alexander Brough (Q15996873)-position held (P39)-member of the State Senate of New York (Q18691526)",
        },
        {
            "role": "assistant",
            "content": "Question: In which country was the author of The Da Vinci Code educated?\n"
            "Nodes mentioned in the question: The Da Vinci Code (Q81689)\n"
            "Answer: United States of America (Q30)\n"
            "Triples used: The Da Vinci Code (Q81689)-author (P50)-Dan Brown (Q7345); Dan Brown (Q7345)-educated at (P69)-Amherst College (Q49165); Amherst College (Q49165)-country (P17)-United States of America (Q30)\n"
            "SPARQL query: SELECT ?answer WHERE {wd:Q81689 wdt:P50 ?person. ?person wdt:P69 ?school. ?school wdt:P17 ?answer}",
        },
        {
            "role": "user",
            "content": "Graph: Jacqueline Kennedy Onassis (Q165421)-child (P40)-John F. Kennedy Jr. (Q316064);Janet Lee Bouvier (Q3097958)-child (P40)-Jacqueline Kennedy Onassis (Q165421);Jacqueline Kennedy Onassis (Q165421)-mother (P25)-Janet Lee Bouvier (Q3097958);Jacqueline Kennedy Onassis (Q165421)-place of birth (P19)-Southampton (Q2631643);Janet Lee Bouvier (Q3097958)-place of death (P20)-Newport (Q54264);Primary (Q2299721)-cast member (P161)-Jacqueline Kennedy Onassis (Q165421);Sue Fuller (Q7634090)-place of death (P20)-Southampton (Q2631643);Sue Fuller (Q7634090)-award received (P166)-Guggenheim Fellowship (Q1316544);Jacqueline Kennedy Onassis (Q165421)-educated at (P69)-University of Paris (Q209842);Elizabeth Pepper (Q5363338)-place of death (P20)-Newport (Q54264);Elizabeth Pepper (Q5363338)-country of citizenship (P27)-United States of America (Q30);Sue Fuller (Q7634090)-country of citizenship (P27)-United States of America (Q30);Janet Lee Bouvier (Q3097958)-country of citizenship (P27)-United States of America (Q30);John F. Kennedy Jr. (Q316064)-country of citizenship (P27)-United States of America (Q30);John F. Kennedy Jr. (Q316064)-mother (P25)-Jacqueline Kennedy Onassis (Q165421);Primary (Q2299721)-country of origin (P495)-United States of America (Q30);Newport (Q54264)-country (P17)-United States of America (Q30);Jacqueline Kennedy Onassis (Q165421)-country of citizenship (P27)-United States of America (Q30);Southampton (Q2631643)-country (P17)-United States of America (Q30);Guggenheim Fellowship (Q1316544)-country (P17)-United States of America (Q30);Dale Morgan (Q5210562)-award received (P166)-Guggenheim Fellowship (Q1316544);Dale Morgan (Q5210562)-country of citizenship (P27)-United States of America (Q30)",
        },
        {
            "role": "assistant",
            "content": "Question: Where did the maternal grandmother of John F. Kennedy Jr. die?\n"
            "Nodes mentioned in the question: John F. Kennedy Jr. (Q316064)\n"
            "Answer: Newport (Q54264)\n"
            "Triples used: John F. Kennedy Jr. (Q316064)-mother (P25)-Jacqueline Kennedy Onassis (Q165421); Jacqueline Kennedy Onassis (Q165421)-mother (P25)-Janet Lee Bouvier (Q3097958); Janet Lee Bouvier (Q3097958)-place of death (P20)-Newport (Q54264)\n"
            "SPARQL query: SELECT ?answer WHERE {wd:Q316064 wdt:P25 ?mother. ?mother wdt:P25 ?grandmother. ?grandmother wdt:P20 ?answer}",
        },
        {
            "role": "user",
            "content": "Graph: Magnus IV of Norway (Q343101)-country of citizenship (P27)-Norway (Q20);Magnus IV of Norway (Q343101)-family (P53)-Hardrada dynasty (Q15715740);Hardrada dynasty (Q15715740)-country (P17)-Norway (Q20);Harald III of Norway (Q203647)-country of citizenship (P27)-Norway (Q20);Harald III of Norway (Q203647)-family (P53)-Hardrada dynasty (Q15715740);Harald III of Norway (Q203647)-religion (P140)-Christianity (Q5043);Henry Gurney (Q2918377)-religion (P140)-Christianity (Q5043);Sigurd the Crusader (Q313005)-family (P53)-Hardrada dynasty (Q15715740);Magnus IV of Norway (Q343101)-father (P22)-Sigurd the Crusader (Q313005);Sigurd the Crusader (Q313005)-followed by (P156)-Magnus IV of Norway (Q343101);Sigurd the Crusader (Q313005)-child (P40)-Magnus IV of Norway (Q343101);Magnus IV of Norway (Q343101)-follows (P155)-Sigurd the Crusader (Q313005);Sigurd the Crusader (Q313005)-country of citizenship (P27)-Norway (Q20);Magnus IV of Norway (Q343101)-place of death (P20)-Hvaler (Q488435);Hvaler (Q488435)-country (P17)-Norway (Q20);Phil Robertson (Q14755515)-religion (P140)-Christianity (Q5043);Hardrada dynasty (Q15715740)-founded by (P112)-Harald III of Norway (Q203647);Magnus Barefoot (Q311996)-family (P53)-Hardrada dynasty (Q15715740);Magnus Barefoot (Q311996)-child (P40)-Sigurd the Crusader (Q313005);Sigurd the Crusader (Q313005)-follows (P155)-Magnus Barefoot (Q311996);Magnus Barefoot (Q311996)-followed by (P156)-Sigurd the Crusader (Q313005);Magnus Barefoot (Q311996)-place of birth (P19)-Norway (Q20);Magnus Barefoot (Q311996)-country of citizenship (P27)-Norway (Q20);Åsta Gudbrandsdatter (Q2543933)-child (P40)-Harald III of Norway (Q203647);Harald IV of Norway (Q260543)-father (P22)-Magnus Barefoot (Q311996);Magnus Barefoot (Q311996)-child (P40)-Harald IV of Norway (Q260543);Sigurd the Crusader (Q313005)-followed by (P156)-Harald IV of Norway (Q260543);Harald IV of Norway (Q260543)-follows (P155)-Sigurd the Crusader (Q313005);Harald IV of Norway (Q260543)-country of citizenship (P27)-Norway (Q20);Judy Glenney (Q6304461)-country of citizenship (P27)-United States of America (Q30);Judy Glenney (Q6304461)-religion (P140)-Christianity (Q5043);Henry Gurney (Q2918377)-country of citizenship (P27)-Malaysia (Q833);Richard Kiel (Q376140)-religion (P140)-Christianity (Q5043)",
        },
        {
            "role": "assistant",
            "content": "Question: What religion did the founder of the family of Magnus IV of Norway follow?\n"
            "Nodes mentioned in the question: Magnus IV of Norway (Q343101)\n"
            "Answer: Christianity (Q5043)\n"
            "Triples used: Magnus IV of Norway (Q343101)-family (P53)-Hardrada dynasty (Q15715740); Hardrada dynasty (Q15715740)-founded by (P112)-Harald III of Norway (Q203647); Harald III of Norway (Q203647)-religion (P140)-Christianity (Q5043)\n"
            "SPARQL query: SELECT ?answer WHERE {wd:Q343101 wdt:P53 ?family. ?family wdt:P112 ?founder. ?founder wdt:P140 ?answer}",
        },
        {
            "role": "user",
            "content": "Graph: The Young Mrs. Eames (Q3990135)-cast member (P161)-Kathlyn Williams (Q2052882);The Young Mrs. Eames (Q3990135)-cast member (P161)-Harold Lockwood (Q127315);Kathlyn Williams (Q2052882)-country of citizenship (P27)-United States of America (Q30);Harold Lockwood (Q127315)-country of citizenship (P27)-United States of America (Q30);The Young Mrs. Eames (Q3990135)-country of origin (P495)-United States of America (Q30);The Young Mrs. Eames (Q3990135)-screenwriter (P58)-Kathlyn Williams (Q2052882);Charles Eyton (Q5077475)-country of citizenship (P27)-United States of America (Q30);Charles Eyton (Q5077475)-spouse (P26)-Kathlyn Williams (Q2052882);Hobart Bosworth (Q1621558)-country of citizenship (P27)-United States of America (Q30);The Young Mrs. Eames (Q3990135)-cast member (P161)-Hobart Bosworth (Q1621558);Pamela Cahill (Q18129157)-country of citizenship (P27)-United States of America (Q30);Lost Illusions (Q3259827)-cast member (P161)-Harold Lockwood (Q127315);Lost Illusions (Q3259827)-country of origin (P495)-United States of America (Q30);Lost Illusions (Q3259827)-cast member (P161)-Lois Weber (Q462440);Lois Weber (Q462440)-country of citizenship (P27)-United States of America (Q30);Coals of Fire (Q3681828)-cast member (P161)-Hobart Bosworth (Q1621558);Coals of Fire (Q3681828)-director (P57)-Hobart Bosworth (Q1621558);Coals of Fire (Q3681828)-country of origin (P495)-United States of America (Q30);The Promise (Q3988858)-cast member (P161)-Harold Lockwood (Q127315);Charles Eyton (Q5077475)-place of burial (P119)-Hollywood Forever Cemetery (Q1624932);Hollywood Forever Cemetery (Q1624932)-country (P17)-United States of America (Q30);Lost Illusions (Q3259827)-cast member (P161)-Phillips Smalley (Q2462046);Phillips Smalley (Q2462046)-country of citizenship (P27)-United States of America (Q30);Phillips Smalley (Q2462046)-spouse (P26)-Lois Weber (Q462440);Lois Weber (Q462440)-spouse (P26)-Phillips Smalley (Q2462046);Kathlyn Williams (Q2052882)-place of death (P20)-Hollywood (Q34006);Charles Eyton (Q5077475)-place of death (P20)-Hollywood (Q34006);Lois Weber (Q462440)-place of death (P20)-Hollywood (Q34006);Phillips Smalley (Q2462046)-place of death (P20)-Hollywood (Q34006);Hollywood (Q34006)-country (P17)-United States of America (Q30);How Men Propose (Q3787188)-director (P57)-Lois Weber (Q462440);How Men Propose (Q3787188)-cast member (P161)-Lois Weber (Q462440);How Men Propose (Q3787188)-screenwriter (P58)-Lois Weber (Q462440)",
        },
        {
            "role": "assistant",
            "content": "Question: Where is the spouse of the screenwriter of The Young Mrs. Eames buried?\n"
            "Nodes mentioned in the question: The Young Mrs. Eames (Q3990135)\n"
            "Answer: Hollywood Forever Cemetery (Q1624932)\n"
            "Triples used: The Young Mrs. Eames (Q3990135)-screenwriter (P58)-Kathlyn Williams (Q2052882); Charles Eyton (Q5077475)-spouse (P26)-Kathlyn Williams (Q2052882); Charles Eyton (Q5077475)-place of burial (P119)-Hollywood Forever Cemetery (Q1624932)\n"
            "SPARQL query: SELECT ?answer WHERE {wd:Q3990135 wdt:P58 ?screenwriter. ?spouse wdt:P26 ?screenwriter. ?spouse wdt:P119 ?answer}",
        },
        {"role": "user", "content": f"Graph: {flattened_graph}"},
    ]


def four_hop_prompt(flattened_graph):
    return [
        {
            "role": "user",
            "content": "Based on the provided set of knowledge graph triples, please generate a question that requires combining the information in exactly four of the provided triples for answering. The answer should correspond to exactly one node in the provided graph, be unique and not ambiguous. Make sure that all four selected triples are required for answering the question. Make sure also that there isn't a smallest set of triples in the graph which is enough to deduce the answer. Respond only with "
            "'Question:' the generated question,\n"
            "'Nodes mentioned in the question:' a semicolon-separated list of the nodes that are explicitly mentioned in the question,\n"
            "'Answer:' The node corresponding to the correct answer with its QID,\n"
            "'Triples used:' a semicolon-separated list of the triples used for answering the question,\n"
            "'SPARQL query:' a SPARQL query to return all answers of the question from the WikiData knowledge base",
        },
        {
            "role": "user",
            "content": "Graph: Girolamo Marciano (Q16560340)-place of birth (P19)-Leverano (Q52149);Girolamo Marciano (Q16560340)-place of death (P20)-Leverano (Q52149);Carmiano (Q52123)-shares border with (P47)-Leverano (Q52149);Biagio Putignano (Q3639357)-place of birth (P19)-Carmiano (Q52123);Copertino (Q52133)-shares border with (P47)-Leverano (Q52149);Leverano (Q52149)-shares border with (P47)-Copertino (Q52133);Salvatore Nobile (Q3945942)-place of birth (P19)-Copertino (Q52133);Copertino (Q52133)-country (P17)-Italy (Q38);Salvatore Nobile (Q3945942)-country of citizenship (P27)-Italy (Q38);Biagio Putignano (Q3639357)-country of citizenship (P27)-Italy (Q38);Carmiano (Q52123)-country (P17)-Italy (Q38);Leverano (Q52149)-country (P17)-Italy (Q38);Girolamo Marciano (Q16560340)-country of citizenship (P27)-Italy (Q38);Biagio Putignano (Q3639357)-occupation (P106)-composer (Q36834);Fabio Mamerto Rivas Santos (Q1390549)-country of citizenship (P27)-Italy (Q38);San Pietro in Lama (Q52189)-shares border with (P47)-Copertino (Q52133);Copertino (Q52133)-shares border with (P47)-San Pietro in Lama (Q52189);San Pietro in Lama (Q52189)-country (P17)-Italy (Q38);Girolamo Marciano (Q16560340)-occupation (P106)-physician (Q39631);Emil August Rikli (Q18616464)-occupation (P106)-physician (Q39631)",
        },
        {
            "role": "assistant",  # Iso-9
            "content": "Question: Which physician was born in a town that borders both Copertino and Carmiano?\n"
            "Nodes mentioned in the question: physician (Q39631); Copertino (Q52133); Carmiano (Q52123)\n"
            "Answer: Girolamo Marciano (Q16560340)\n"
            "Triples used: Girolamo Marciano (Q16560340)-occupation (P106)-physician (Q39631); Girolamo Marciano (Q16560340)-place of birth (P19)-Leverano (Q52149); Carmiano (Q52123)-shares border with (P47)-Leverano (Q52149); Leverano (Q52149)-shares border with (P47)-Copertino (Q52133)\n"
            "SPARQL query: SELECT ?answer WHERE {?answer wdt:P106 wd:Q39631. ?answer wdt:P19 ?place. wd:Q52123 wdt:P47 ?place. wd:Q52133 wdt:P47 ?place}",
        },
        {
            "role": "user",
            "content": "Graph: Augustus Uthwatt, Baron Uthwatt (Q2277548)-country of citizenship (P27)-United Kingdom (Q145);Paul McLeod (Q7152404)-country of citizenship (P27)-United Kingdom (Q145);Balliol College (Q805285)-country (P17)-United Kingdom (Q145);Augustus Uthwatt, Baron Uthwatt (Q2277548)-educated at (P69)-Balliol College (Q805285);Paul McLeod (Q7152404)-member of sports team (P54)-Albion Rovers F.C. (Q656134);Augustus Uthwatt, Baron Uthwatt (Q2277548)-occupation (P106)-judge (Q16533);Paul McLeod (Q7152404)-occupation (P106)-association football player (Q937857);Reginald H. Jones (Q7308721)-country of citizenship (P27)-United Kingdom (Q145);Reginald H. Jones (Q7308721)-family name (P734)-Jones (Q59853);Cho Byung-young (Q5103330)-occupation (P106)-association football player (Q937857);Denis Healey (Q332360)-country of citizenship (P27)-United Kingdom (Q145);Denis Healey (Q332360)-educated at (P69)-Balliol College (Q805285);Augustus Uthwatt, Baron Uthwatt (Q2277548)-place of birth (P19)-Ballarat (Q17856)",
        },
        {
            "role": "assistant",  # Iso-18
            "content": "Question: Which judge, who was born in Ballarat, was educated at an institution located in the United Kingdom?\n"
            "Nodes mentioned in the question: judge (Q16533); Ballarat (Q17856); United Kingdom (Q145)\n"
            "Answer: Augustus Uthwatt, Baron Uthwatt (Q2277548)\n"
            "Triples used: Augustus Uthwatt, Baron Uthwatt (Q2277548)-place of birth (P19)-Ballarat (Q17856); Augustus Uthwatt, Baron Uthwatt (Q2277548)-occupation (P106)-judge (Q16533); Augustus Uthwatt, Baron Uthwatt (Q2277548)-educated at (P69)-Balliol College (Q805285); Balliol College (Q805285)-country (P17)-United Kingdom (Q145)\n"
            "SPARQL query: SELECT ?answer WHERE {?answer wdt:P19 wd:Q17856. ?answer wdt:P106 wd:Q16533. ?answer wdt:P69 ?place. ?place wdt:P17 wd:Q145.}",
        },
        {
            "role": "user",
            "content": "Graph: Ricciotti Garibaldi jr (Q15292901)-country of citizenship (P27)-Italy (Q38);Giuseppe Tuntar (Q16561082)-country of citizenship (P27)-Italy (Q38);Ricciotti Garibaldi jr (Q15292901)-occupation (P106)-spy (Q9352089);Lucio Campiani (Q3838822)-country of citizenship (P27)-Italy (Q38);Kees van Brink (Q2780540)-occupation (P106)-spy (Q9352089);Italy (Q38)-capital (P36)-Rome (Q220);Rome (Q220)-country (P17)-Italy (Q38);Rome (Q220)-capital of (P1376)-Italy (Q38);Ricciotti Garibaldi jr (Q15292901)-place of death (P20)-Rome (Q220);Ricciotti Garibaldi jr (Q15292901)-place of birth (P19)-Rome (Q220);Magda Fontanges (Q3276622)-occupation (P106)-spy (Q9352089);Magda Fontanges (Q3276622)-occupation (P106)-journalist (Q1930187);Giuseppe Tuntar (Q16561082)-occupation (P106)-journalist (Q1930187);Ricciotti Garibaldi (Q1236827)-place of death (P20)-Rome (Q220);Ricciotti Garibaldi (Q1236827)-country of citizenship (P27)-Italy (Q38);Ricciotti Garibaldi (Q1236827)-child (P40)-Ricciotti Garibaldi jr (Q15292901);Ricciotti Garibaldi jr (Q15292901)-father (P22)-Ricciotti Garibaldi (Q1236827);Magda Fontanges (Q3276622)-occupation (P106)-actor (Q33999);Straziami, ma di baci saziami (Q2904749)-filming location (P915)-Rome (Q220);Straziami, ma di baci saziami (Q2904749)-country of origin (P495)-Italy (Q38)",
        },
        {
            "role": "assistant",  # Iso-8
            "content": "Question: What is the country of citizenship of the spy whose father died in Rome?\n"
            "Nodes mentioned in the question: spy (Q9352089); Rome (Q220)\n"
            "Answer: Italy (Q38)\n"
            "Triples used: Ricciotti Garibaldi jr (Q15292901)-occupation (P106)-spy (Q9352089); Ricciotti Garibaldi jr (Q15292901)-father (P22)-Ricciotti Garibaldi (Q1236827); Ricciotti Garibaldi (Q1236827)-place of death (P20)-Rome (Q220); Ricciotti Garibaldi jr (Q15292901)-country of citizenship (P27)-Italy (Q38)\n"
            "SPARQL query: SELECT ?answer WHERE {?person wdt:P106 wd:Q9352089. ?person wdt:P22 ?father. ?father wdt:P20 wd:Q220. ?person wdt:P27 ?answer}",
        },
        {
            "role": "user",
            "content": "Graph: Maurice (Q432088)-production company (P272)-Merchant Ivory Productions (Q2407366);Surviving Picasso (Q1680069)-production company (P272)-Merchant Ivory Productions (Q2407366);Shakespeare Wallah (Q3481272)-production company (P272)-Merchant Ivory Productions (Q2407366);The Golden Bowl (Q1774070)-production company (P272)-Merchant Ivory Productions (Q2407366);Shakespeare Wallah (Q3481272)-cast member (P161)-Felicity Kendal (Q2603654);We're Back! A Dinosaur's Story (Q1575362)-cast member (P161)-Felicity Kendal (Q2603654);Savages (Q3954950)-production company (P272)-Merchant Ivory Productions (Q2407366);Surviving Picasso (Q1680069)-screenwriter (P58)-Ruth Prawer Jhabvala (Q235759);Shakespeare Wallah (Q3481272)-screenwriter (P58)-Ruth Prawer Jhabvala (Q235759);The Golden Bowl (Q1774070)-screenwriter (P58)-Ruth Prawer Jhabvala (Q235759);We're Back! A Dinosaur's Story (Q1575362)-cast member (P161)-Julia Child (Q214477);Heat and Dust (Q2549369)-screenwriter (P58)-Ruth Prawer Jhabvala (Q235759);Heat and Dust (Q2549369)-production company (P272)-Merchant Ivory Productions (Q2407366);Heat and Dust (Q2549369)-based on (P144)-Heat and Dust (Q673521);Heat and Dust (Q673521)-author (P50)-Ruth Prawer Jhabvala (Q235759);Julia Child (Q214477)-member of (P463)-American Academy of Arts and Sciences (Q463303);Ruth Prawer Jhabvala (Q235759)-child (P40)-Renana Jhabvala (Q4151026);Renana Jhabvala (Q4151026)-mother (P25)-Ruth Prawer Jhabvala (Q235759);Heat and Dust (Q673521)-original language of film or TV show (P364)-English (Q1860);Savages (Q3954950)-original language of film or TV show (P364)-English (Q1860);Maurice (Q432088)-original language of film or TV show (P364)-English (Q1860);Surviving Picasso (Q1680069)-original language of film or TV show (P364)-English (Q1860);Heat and Dust (Q2549369)-original language of film or TV show (P364)-English (Q1860);India (Q668)-official language (P37)-English (Q1860);Edward the Seventh (Q3048737)-original language of film or TV show (P364)-English (Q1860);Ruth Prawer Jhabvala (Q235759)-languages spoken, written or signed (P1412)-English (Q1860);Simon Callow (Q731187)-languages spoken, written or signed (P1412)-English (Q1860);Jane Austen in Manhattan (Q1405420)-original language of film or TV show (P364)-English (Q1860);Jane Austen in Manhattan (Q1405420)-screenwriter (P58)-Ruth Prawer Jhabvala (Q235759)",
        },
        {
            "role": "assistant",  # Iso-7
            "content": "Which film produced by Merchant Ivory Productions is based on a work authored by the mother of Renana Jhabvala?\n"
            "Nodes mentioned in the question: Merchant Ivory Productions (Q2407366); Renana Jhabvala (Q4151026)\n"
            "Answer: Heat and Dust (Q2549369)\n"
            "Triples used: Heat and Dust (Q2549369)-production company (P272)-Merchant Ivory Productions (Q2407366); Heat and Dust (Q2549369)-based on (P144)-Heat and Dust (Q673521); Renana Jhabvala (Q4151026)-mother (P25)-Ruth Prawer Jhabvala (Q235759); Heat and Dust (Q673521)-author (P50)-Ruth Prawer Jhabvala (Q235759)\n"
            "SPARQL query: SELECT ?answer WHERE { ?answer wdt:P272 wd:Q2407366; wdt:P144 ?work. ?work wdt:P50 ?author. wd:Q4151026 wdt:P25 ?author.}",
        },
        {
            "role": "user",
            "content": "Graph: Beuron Archabbey (Q450038)-architectural style (P149)-baroque architecture (Q840829);Willibrord Benzler (Q91025)-place of burial (P119)-Beuron Archabbey (Q450038);St.-Martin-Kirche (Q1121842)-architectural style (P149)-baroque architecture (Q840829);Beuron Archabbey (Q450038)-country (P17)-Germany (Q183);Willibrord Benzler (Q91025)-country of citizenship (P27)-Germany (Q183);St.-Martin-Kirche (Q1121842)-country (P17)-Germany (Q183);Franz Vogel (Q1449330)-country of citizenship (P27)-Germany (Q183);Franz Vogel (Q1449330)-place of birth (P19)-Berlin (Q64);Willibrord Benzler (Q91025)-religious order (P611)-Benedictines (Q131132);Franz Vogel (Q1449330)-place of death (P20)-Berlin (Q64);Berlin (Q64)-located in the administrative territorial entity (P131)-Germany (Q183);Berlin (Q64)-country (P17)-Germany (Q183);Germany (Q183)-capital (P36)-Berlin (Q64);Germany (Q183)-contains administrative territorial entity (P150)-Berlin (Q64);Berlin (Q64)-capital of (P1376)-Germany (Q183);Konrad Zuse (Q60093)-country of citizenship (P27)-Germany (Q183);Konrad Zuse (Q60093)-place of birth (P19)-Berlin (Q64);Willibrord Benzler (Q91025)-work location (P937)-Metz (Q22690);Willibrord Benzler (Q91025)-notable work (P800)-Erinnerungen aus meinem Leben (Q15624681);Erinnerungen aus meinem Leben (Q15624681)-author (P50)-Willibrord Benzler (Q91025);Erinnerungen aus meinem Leben (Q15624681)-editor (P98)-Pius Bihlmeyer (Q15627474);Erinnerungen aus meinem Leben (Q15624681)-author (P50)-Pius Bihlmeyer (Q15627474);Willibrord Benzler (Q91025)-position held (P39)-titular bishop (Q948657);cultural heritage monument in Germany (Q11691318)-country (P17)-Germany (Q183);St.-Martin-Kirche (Q1121842)-heritage designation (P1435)-cultural heritage monument in Germany (Q11691318);Albert Thiery (Q16026552)-place of birth (P19)-Metz (Q22690);Pius Bihlmeyer (Q15627474)-place of death (P20)-Freiburg im Breisgau (Q2833);Franz Vogel (Q1449330)-occupation (P106)-film producer (Q3282637);Otto Brucks (Q17590782)-place of death (P20)-Metz (Q22690);Hans Lembke (Q1581028)-place of birth (P19)-Freiburg im Breisgau (Q2833);Hans Lembke (Q1581028)-country of citizenship (P27)-Germany (Q183);Pius Bihlmeyer (Q15627474)-place of birth (P19)-Aulendorf (Q505459);Aulendorf (Q505459)-country (P17)-Germany (Q183);Karl Aloys Königsegg-Aulendorf (Q1730273)-place of birth (P19)-Aulendorf (Q505459);Karl Aloys Königsegg-Aulendorf (Q1730273)-country of citizenship (P27)-Germany (Q183)",
        },
        {
            "role": "assistant",  # Iso-10
            "content": "Who edited the notable work of the Benedictine buried at Beuron Archabbey?\n"
            "Nodes mentioned in the question: Beuron Archabbey (Q450038); Benedictines (Q131132)\n"
            "Answer: Pius Bihlmeyer (Q15627474)\n"
            "Triples used: Willibrord Benzler (Q91025)-religious order (P611)-Benedictines (Q131132); Willibrord Benzler (Q91025)-place of burial (P119)-Beuron Archabbey (Q450038); Willibrord Benzler (Q91025)-notable work (P800)-Erinnerungen aus meinem Leben (Q15624681); Erinnerungen aus meinem Leben (Q15624681)-editor (P98)-Pius Bihlmeyer (Q15627474)\n"
            "SPARQL query: SELECT ?answer WHERE { ?person wdt:P611 wd:Q131132; wdt:P119 wd:Q450038; wdt:P800 ?work. ?work wdt:P98 ?answer. }",
        },
        {
            "role": "user",
            "content": "Graph: Wilhelm Wirth (Q102614)-country of citizenship (P27)-Germany (Q183);National Socialist German Workers' Party (Q7320)-country (P17)-Germany (Q183);Wilhelm Wirth (Q102614)-member of political party (P102)-National Socialist German Workers' Party (Q7320);Wilhelm Wirth (Q102614)-place of birth (P19)-Leipzig (Q2079);Wilhelm Wirth (Q102614)-work location (P937)-Leipzig (Q2079);Leipzig (Q2079)-country (P17)-Germany (Q183);Wilhelm Wirth (Q102614)-place of death (P20)-Amberg (Q7042);Amberg (Q7042)-country (P17)-Germany (Q183);Roland Asch (Q78021)-country of citizenship (P27)-Germany (Q183);Wilhelm Wirth (Q102614)-occupation (P106)-psychologist (Q212980);Ernst Baumgarten (Q17453073)-place of birth (P19)-Leipzig (Q2079);Ernst Baumgarten (Q17453073)-place of death (P20)-Leipzig (Q2079);Ernst Baumgarten (Q17453073)-country of citizenship (P27)-Germany (Q183);Roland Asch (Q78021)-place of birth (P19)-Tübingen (Q3806);Tübingen (Q3806)-country (P17)-Germany (Q183);James Steuart (Q257107)-work location (P937)-Tübingen (Q3806);Roland Asch (Q78021)-occupation (P106)-racing driver (Q378622);Uli Schwarz (Q1578187)-work location (P937)-Tübingen (Q3806);Uli Schwarz (Q1578187)-place of death (P20)-Tübingen (Q3806);Uli Schwarz (Q1578187)-country of citizenship (P27)-Germany (Q183)",
        },
        {
            "role": "assistant",  # Iso-12
            "content": "Question: What is the place of death of the psychologist working in Leipzig that was a member of the National Socialist German Workers' Party?\n"
            "Nodes mentioned in the question: psychologist (Q212980); Leipzig (Q2079); National Socialist German Workers' Party (Q7320)\n"
            "Answer: Amberg (Q7042)\n"
            "Triples used: Wilhelm Wirth (Q102614)-work location (P937)-Leipzig (Q2079); Wilhelm Wirth (Q102614)-member of political party (P102)-National Socialist German Workers' Party (Q7320); Wilhelm Wirth (Q102614)-occupation (P106)-psychologist (Q212980); Wilhelm Wirth (Q102614)-place of death (P20)-Amberg (Q7042)\n"
            "SPARQL query: SELECT ?answer WHERE {?person wdt:P20 ?answer. ?person wdt:P937 wd:Q2079. ?person wdt:P102 wd:Q7320. ?person wdt:P106 wd:Q212980}",
        },
        {"role": "user", "content": f"Graph: {flattened_graph}"},
    ]


def paraphrase_question(question):
    return [
        {
            "role": "user",
            "content": "Given a question, please paraphrase it in a more natural form. Make sure that the meaning of the question and its answer remain unaltered. Make sure that all the entities that are explicitly mentioned in the original question still appear in the paraphrased question. Only output the paraphrased question.",
        },
        {
            "role": "user",
            "content": "Question: What is the country of origin of the film directed by Ariane Mnouchkine that has French as its original language?",
        },
        {
            "role": "assistant",
            "content": "Which country originated the movie by Ariane Mnouchkine with French as original language?",
        },
        {
            "role": "user",
            "content": "Question: In which league does the team whose home venue is shared with Altay S.K. and whose head coach is Cihat Arslan play?",
        },
        {
            "role": "assistant",
            "content": "Which league is the team coached by Cihat Arslan, and that play home games in the same venue as Altay S.K., competing in?",
        },
        {"role": "user", "content": f"Question: {question}"},
    ]


def evaluate_complete_prompt(question, knowledge_triplets):
    # Adapted from the prompts in https://github.com/GasolSun36/ToG/blob/main/ToG/prompt_list.py
    return [
        {
            "role": "user",
            "content": "Given a question and the associated retrieved knowledge graph triplets (subject, relation, object), you are asked to answer the question by combining the information contained in the triplets with your own knowledge. Notice that a knowledge triple like (a, student, b) means that b is a student of a (not viceversa!). Similarly, (a, child, b) means that b is a child of a. Similarly (a, doctoral advisor, b) means that b is the doctoral advisor of a. Make your responses as concise as possible, with your final answer between curly brackets {}.",
        },
        {
            "role": "user",
            "content": "Question: Yamaji Motoharu was a general in an army of which Empire?\n"
            "Knowledge Triplets: Imperial Japanese Army, allegiance, Empire of Japan\n"
            "Yamaji Motoharu, member of, Imperial Japanese Army\n"
            "Yamaji Motoharu, military rank, general\n",
        },
        {
            "role": "assistant",
            "content": "Answer: The answer to the question is {Empire of Japan}.",
        },
        {
            "role": "user",
            "content": "Question: Which member of the British Army took part in the Summer Olympics in both 1956 and 1960?\n"
            "Knowledge Triplets: Nik Stuart, military branch, British Army\n"
            "Nik Stuart, participant in, 1956 Summer Olympics\n"
            "1956 Summer Olympics, place of event, Melbourne\n"
            "Nik Stuart, participant in, 1960 Summer Olympics\n",
        },
        {"role": "assistant", "content": "Answer: The answer is {Nik Stuart}."},
        {
            "role": "user",
            "content": f"Question: {question}\n Knowledge Triplets: {knowledge_triplets}",
        },
    ]


def cot_prompt(question):
    # Adapted from the prompts in https://github.com/GasolSun36/ToG/blob/main/ToG/prompt_list.py
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
    # Adapted from the prompts in https://github.com/GasolSun36/ToG/blob/main/ToG/prompt_list.py
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
