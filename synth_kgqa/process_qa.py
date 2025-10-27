# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import json
import os.path as osp
import pickle
import re
import time
from collections import Counter

import numpy as np
from process_utils import (
    build_n_hops_mapper,
    check_redundant_subquery,
    check_template,
    parse_llm_label,
)
from rdflib import Graph, Namespace
from SPARQLWrapper import JSON, SPARQLWrapper
from tqdm import tqdm

from synth_kgqa import llm, parse
from synth_kgqa.prompts import paraphrase_question


def main(args):
    llm_api = llm.LLMAPI(args.api_name, model=args.llm)

    with open(osp.join(args.qa_path, "samples.pkl"), "rb") as f:
        samples = pickle.load(f)
    with open(osp.join(args.qa_path, "llm_outputs.pkl"), "rb") as f:
        llm_out = pickle.load(f)

    edge_ids = np.load(osp.join(args.kg_path, "edge_ids.npy"))
    relation_types = np.load(osp.join(args.kg_path, "relation_types.npy"))
    node_labels = np.load(osp.join(args.kg_path, "node_labels.npy"), allow_pickle=True)
    relation_labels = np.load(
        osp.join(args.kg_path, "relation_labels.npy"), allow_pickle=True
    )
    node_qids = np.load(osp.join(args.kg_path, "node_ids.npy"), allow_pickle=True)
    relation_pids = np.load(
        osp.join(args.kg_path, "relation_ids.npy"),
        allow_pickle=True,
    )
    qid_lookup = {q: i for i, q in enumerate(node_qids)}
    pid_lookup = {p: i for i, p in enumerate(relation_pids)}

    valid_idx = []
    llm_questions_list = []
    llm_question_nodes_list = []
    llm_answer_list = []
    llm_triples_list = []
    llm_sparql_list = []
    sample_list = []

    sparql = SPARQLWrapper(
        "https://query.wikidata.org/sparql",
        agent="user_id",
    )
    sparql.setReturnFormat(JSON)

    # rdflib to run SPARQL queries on ogbl-wikikg2
    wikikg2 = Graph()
    wd = Namespace("http://example.org/entities/")
    wdt = Namespace("http://example.org/relations/")
    for i in tqdm(range(edge_ids.shape[1])):
        s = wd[node_qids[edge_ids[0, i]]]
        p = wdt[relation_pids[relation_types[i]]]
        o = wd[node_qids[edge_ids[1, i]]]
        wikikg2.add((s, p, o))
    wikikg2.bind("wd", wd)
    wikikg2.bind("wdt", wdt)

    n_hops_mapper = build_n_hops_mapper()

    for i, out in enumerate(llm_out):
        question_search = re.search(r"Question:(.*)$", out, re.MULTILINE)
        question_node_ids_search = re.search(
            r"Nodes mentioned in the question:(.*)$", out, re.MULTILINE
        )
        answer_search = re.search(r"Answer:(.*)$", out, re.MULTILINE)
        triples_search = re.search(r"Triples used:(.*)$", out, re.MULTILINE)
        sparql_search = re.search(
            r"SPARQL query:(.*)$", out.replace("\n", ""), re.MULTILINE
        )

        if (
            question_search
            and question_node_ids_search
            and answer_search
            and triples_search
            and sparql_search
        ):
            llm_questions_list.append(question_search.group(1).strip())
            llm_question_nodes_list.append(
                question_node_ids_search.group(1).strip().split(";")
            )
            llm_answer_list.append(answer_search.group(1).strip())
            llm_triples_list.append(triples_search.group(1).strip().split(";"))
            llm_sparql_list.append(sparql_search.group(1).strip())
            sample_list.append(samples[i])
            valid_idx.append(i)

    qa = []
    invalid_label_format = []
    no_node_match = []
    no_triple_match = []
    invalid_sparql = []
    wrong_sparql_answer = []
    exceed_sparql_answer = []
    invalid_template = []
    redundant_info = []

    pbar = tqdm(
        enumerate(
            zip(
                sample_list,
                llm_questions_list,
                llm_question_nodes_list,
                llm_answer_list,
                llm_triples_list,
                llm_sparql_list,
            )
        ),
        total=len(sample_list),
    )

    for n, (
        sample,
        llm_question,
        llm_question_nodes,
        llm_answer,
        llm_triples,
        llm_sparql,
    ) in pbar:
        if n > 0:
            pbar.set_description(f"Acceptance rate: {(100 * len(qa)/n):.1f}%")
        # check answer format
        answ_parse = parse_llm_label(
            llm_answer, node_labels, relation_labels, qid_lookup, pid_lookup
        )
        if len(answ_parse) == 0:
            invalid_label_format.append(n)
            continue
        else:
            answer_lab, answer_qid, answer_id = answ_parse

        # check question nodes format
        q_node_qids = []
        q_node_ids = []
        for q_node in llm_question_nodes:
            q_node_parse = parse_llm_label(
                q_node, node_labels, relation_labels, qid_lookup, pid_lookup
            )
            if len(q_node_parse) == 0:
                break
            else:
                q_node_qids.append(q_node_parse[-2])
                q_node_ids.append(q_node_parse[-1])
        if len(q_node_ids) != len(llm_question_nodes):
            invalid_label_format.append(valid_idx[n])
            continue

        # ensure all nodes in `llm_question_nodes` are actual node in the subgraph
        if (
            not (np.array(q_node_ids)[:, None] == sample["edge_ids"].flatten())
            .any(-1)
            .all()
        ):
            no_node_match.append(valid_idx[n])
            continue

        # check triples format
        triples = []
        for llmtriple in llm_triples:
            triple_split = re.split(r"\)\s?-", llmtriple)
            if len(triple_split) != 3 or min([len(x) for x in triple_split]) < 1:
                break
            triple_split = [x + ")" if x[-1] != ")" else x for x in triple_split]
            triple = []
            for split in triple_split:
                triples_split_parse = parse_llm_label(
                    split, node_labels, relation_labels, qid_lookup, pid_lookup
                )
                if len(triples_split_parse) == 0:
                    break
                else:
                    triple.append(triples_split_parse[-1])
            if len(triple) == 3:
                triples.append(triple)
            else:
                break
        if len(llm_triples) != len(triples):
            invalid_label_format.append(valid_idx[n])
            continue
        # discard duplicated triples, keep ordering
        triples = np.array(triples)
        triples = triples[np.sort(np.unique(triples, axis=0, return_index=True)[1])]

        # ensure all triples in `llm_triples_list` are actual triples in the subgraph
        subgraph = np.stack(
            [sample["edge_ids"][0], sample["relation_types"], sample["edge_ids"][1]]
        ).T
        if not (triples[:, None] == subgraph[None, :]).all(-1).any(-1).all():
            no_triple_match.append(valid_idx[n])
            continue
        triples = triples.tolist()

        # ensure all entities mentioned in SPARQL query are seed nodes
        qids_in_sparql = re.findall(r"wd:([Q0-9]+)", llm_sparql)
        contained = True
        for seed in q_node_qids:
            if seed not in qids_in_sparql:
                contained = False
                break

        if contained:
            for seed in qids_in_sparql:
                if seed not in q_node_qids:
                    contained = False
                    break

        if (not contained) or "rdfs" in llm_sparql:
            invalid_sparql.append(valid_idx[n])
            continue

        # retrieve all answers from Wikidata
        sparql.setQuery(llm_sparql)
        try:
            time.sleep(0.002)  # to avoid exceeding Wikidata query limits
            ret = sparql.queryAndConvert()
            sparql_qids = [
                result["answer"]["value"].split("/")[-1]
                for result in ret["results"]["bindings"]
            ]
            sparql_qids = np.unique(np.array(sparql_qids)).tolist()
        except Exception as e:
            sparql_qids = []

        if answer_qid not in sparql_qids:
            wrong_sparql_answer.append(valid_idx[n])
            continue
        if len(sparql_qids) > args.max_num_answers:
            exceed_sparql_answer.append(valid_idx[n])
            continue

        # check isomorphism class of answer subgraph
        valid_template = check_template(triples, q_node_ids, answer_id)
        if valid_template:
            construct_query = "CONSTRUCT WHERE" + llm_sparql.split("WHERE")[-1]
            try:
                # retrieve full subgraph (over all possible answers)
                sparql.setQuery(construct_query)
                ret = sparql.queryAndConvert()
                full_graph = [
                    (
                        result["subject"]["value"].split("/")[-1],
                        result["predicate"]["value"].split("/")[-1],
                        result["object"]["value"].split("/")[-1],
                    )
                    for result in ret["results"]["bindings"]
                ]
            except Exception as e:
                invalid_sparql.append(valid_idx[n])
                continue

            if len(full_graph) < len(triples):
                invalid_sparql.append(valid_idx[n])
                continue

            # retrieve from ogbl-wikikg2
            qres = wikikg2.query(llm_sparql)
            sparql_qids_wikikg2 = [x.answer.split("/")[-1] for x in qres]
            qres = wikikg2.query(construct_query)
            full_graph_wikikg2 = [
                (
                    str(t[0]).split("/")[-1],
                    str(t[1]).split("/")[-1],
                    str(t[2]).split("/")[-1],
                )
                for t in qres
            ]

            # check for redundant information in question
            minimal_template, minimal_subgraphs, minimal_queries = (
                check_redundant_subquery(
                    sparql,
                    triples,
                    answer_id,
                    sparql_qids,
                    q_node_ids,
                    node_qids,
                    relation_pids,
                )
            )
            if minimal_template is None:
                invalid_template.append(valid_idx[n])
                continue
            redundant = minimal_template != valid_template
            if redundant:
                redundant_info.append(valid_idx[n])
            qa.append(
                {
                    "question": llm_question,
                    "paraphrased_question": llm_api(
                        prompt=paraphrase_question(llm_question)
                    ),
                    "seed_nodes": [x.strip() for x in llm_question_nodes],
                    "seed_nodes_id": q_node_ids,
                    "answer_node": llm_answer,
                    "answer_node_id": answer_id,
                    "sparql_query": llm_sparql,
                    "all_answers": sparql_qids,
                    "all_answers_wikikg2": sparql_qids_wikikg2,
                    "answer_subgraph": triples,
                    "full_subgraph": full_graph,
                    "full_subgraph_wikikg2": full_graph_wikikg2,
                    "n_hops": n_hops_mapper[valid_template],
                    "graph_template": valid_template,
                    "answer_triples": llm_triples,
                    "redundant": redundant,
                    "minimal_graph_template": minimal_template,
                    "minimal_answer_subgraph_per_seed": minimal_subgraphs,
                    "minimal_query_per_seed": minimal_queries,
                    "source": samples[valid_idx[n]].get(
                        "expansion_source", f"{args.qa_path}-{valid_idx[n]}"
                    ),
                }
            )
        else:
            invalid_template.append(valid_idx[n])

    print(f"Starting questions: {len(llm_out)}")
    print(f"Valid LLM outputs: {len(llm_questions_list)}")
    print(f"Invalid LLM labels format: {len(invalid_label_format)}")
    print(f"No node match: {len(no_node_match)}")
    print(f"No triple match: {len(no_triple_match)}")
    print(f"Invalid SPARQL query: {len(invalid_sparql)}")
    print(f"Wrong SPARQL answer: {len(wrong_sparql_answer)}")
    print(f"Exceeding answers: {len(exceed_sparql_answer)}")
    print(f"Invalid graph template: {len(invalid_template)}")
    print(f"Final valid questions: {len(qa)}")
    print(f"--of which, with redundant info: {len(redundant_info)}")

    log = [
        {
            "starting_questions": len(llm_out),
            "invalid_llm_out": [i for i in range(len(llm_out)) if i not in valid_idx],
            "invalid_label_format": invalid_label_format,
            "no_node_match": no_node_match,
            "no_triple_match": no_triple_match,
            "invalid_sparql": invalid_sparql,
            "no_answer_match": wrong_sparql_answer,
            "exceeding_answer": exceed_sparql_answer,
            "invalid_graph_template": invalid_template,
        }
    ]

    template_counts = Counter([x["graph_template"] for x in qa])
    print(f"Template counts: {template_counts}")

    template_counts = Counter([x["graph_template"] for x in qa if not x["redundant"]])
    print(f"Template counts (no redundant info): {template_counts}")

    with open(osp.join(args.qa_path, "processed_qa.json"), "w") as f:
        json.dump(qa, f)
    with open(osp.join(args.qa_path, "filter_log.json"), "w") as f_log:
        json.dump(log, f_log)


if __name__ == "__main__":
    args = parse.parse_processing_args()
    main(args)
