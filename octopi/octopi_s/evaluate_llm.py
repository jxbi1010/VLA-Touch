import argparse
import json
import math
import natsort
import os
import re


def evaluate_ranking(data):
    def get_rankings(text):
        text = text.split("decreasing")[1:]
        try:
            for i, txt in enumerate(text):
                text[i] = text[i].replace(">=", "=").replace(">", ",")
                text[i] = re.sub(r"[^\d.,=]", "", text[i]).strip(".")
            hardness_order = [i.strip() for i in text[0].split(",")]
            roughness_order = [i.strip() for i in text[1].split(",")]
            hardness_ranks = {}
            roughness_ranks = {}
            for i in range(len(hardness_order)):
                if "=" in hardness_order[i]:
                    for j in hardness_order[i].split("="):
                        hardness_ranks[j] = i
                else:
                    hardness_ranks[hardness_order[i]] = i
            for i in range(len(roughness_order)):
                if "=" in roughness_order[i]:
                    for j in roughness_order[i].split("="):
                        roughness_ranks[j] = i
                else:
                    roughness_ranks[roughness_order[i]] = i
        except:
            return None, None
        return hardness_ranks, roughness_ranks

    property_order_results = {
        "no_ranking": 0,
        "invalid_ranking_count": 0
    }
    for d in data:
        generation = d["final_generation"]
        answer = d["final_true_answer"]
        if "decreasing" not in d["final_true_answer"]:
            continue
        if "decreasing" not in d["final_generation"]:
            property_order_results["no_ranking"] += 1
            continue
        generation_hardness_order, generation_roughness_order = get_rankings(generation)
        try:
            num_hardness_objects = len(generation_hardness_order)
            num_roughness_objects = len(generation_roughness_order)
        except TypeError:
            property_order_results["invalid_ranking_count"] += 1
            print(generation)
            continue
        answer_hardness_order, answer_roughness_order = get_rankings(answer)
        if natsort.natsorted(generation_hardness_order) != natsort.natsorted(answer_hardness_order) or natsort.natsorted(generation_roughness_order) != natsort.natsorted(answer_roughness_order):
            property_order_results["invalid_ranking_count"] += 1
        else:
            pairwise_count = sum([i for i in range(num_hardness_objects)])
            if num_hardness_objects not in property_order_results.keys():
                property_order_results[num_hardness_objects] = {
                    "pairwise_count": pairwise_count,
                    "hardness_pairwise_correct": 0,
                    "roughness_pairwise_correct": 0,
                    "count": 1,
                    "hardness_correct": 0,
                    "roughness_correct": 0
                }
            else:
                property_order_results[num_hardness_objects]["pairwise_count"] += pairwise_count
                property_order_results[num_hardness_objects]["count"] += 1
            for i in natsort.natsorted(generation_hardness_order):
                for j in natsort.natsorted(generation_hardness_order):
                    if j <= i:
                        continue
                    else:
                        if generation_hardness_order[i] - generation_hardness_order[j] < 0 and answer_hardness_order[i] - answer_hardness_order[j] < 0:
                            property_order_results[num_hardness_objects]["hardness_pairwise_correct"] += 1
                        elif generation_hardness_order[i] - generation_hardness_order[j] > 0 and answer_hardness_order[i] - answer_hardness_order[j] > 0:
                            property_order_results[num_hardness_objects]["hardness_pairwise_correct"] += 1
                        elif generation_hardness_order[i] - generation_hardness_order[j] == 0 and answer_hardness_order[i] - answer_hardness_order[j] == 0:
                            property_order_results[num_hardness_objects]["hardness_pairwise_correct"] += 1
            if generation_hardness_order == answer_hardness_order:
                property_order_results[num_hardness_objects]["hardness_correct"] += 1
            for i in natsort.natsorted(generation_roughness_order):
                for j in natsort.natsorted(generation_roughness_order):
                    if j <= i:
                        continue
                    else:
                        if generation_roughness_order[i] - generation_roughness_order[j] < 0 and answer_roughness_order[i] - answer_roughness_order[j] < 0:
                            property_order_results[num_roughness_objects]["roughness_pairwise_correct"] += 1
                        elif generation_roughness_order[i] - generation_roughness_order[j] > 0 and answer_roughness_order[i] - answer_roughness_order[j] > 0:
                            property_order_results[num_roughness_objects]["roughness_pairwise_correct"] += 1
                        elif generation_roughness_order[i] - generation_roughness_order[j] == 0 and answer_roughness_order[i] - answer_roughness_order[j] == 0:
                            property_order_results[num_roughness_objects]["roughness_pairwise_correct"] += 1
            if generation_roughness_order == answer_roughness_order:
                property_order_results[num_roughness_objects]["roughness_correct"] += 1
    accuracy = {i: {} for i in property_order_results.keys() if type(i) == int and i != 1}
    for cnt, result in property_order_results.items():
        if cnt == 1:
            # Only one object
            continue
        elif type(cnt) == str:
            continue
        else:
            accuracy[cnt] = {
                "hardness_pairwise": result["hardness_pairwise_correct"] / result["pairwise_count"],
                "roughness_pairwise": result["roughness_pairwise_correct"] / result["pairwise_count"],
                "hardness": result["hardness_correct"] / result["count"],
                "roughness": result["roughness_correct"] / result["count"]
            }
    return accuracy, property_order_results


def evaluate_reasoning(data):
    correct, cnt = 0, 0
    for d in data:
        generation = d["final_generation"].replace("*", "").split("Answer: ")[-1][0]
        answer = d["final_true_answer"][0]
        print(f"Answer: {answer}; Generation: {generation}")
        if generation == answer:
            correct += 1
        cnt += 1
    accuracy = correct / cnt
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_preds_path', help='LLM prediction JSON file')
    args = parser.parse_args()
    with open(args.llm_preds_path, "r") as f:
        data = json.load(f)
        f.close()

    if "/reason/" in args.llm_preds_path:
        # Scenario reasoning
        reasoning_accuracy = evaluate_reasoning(data)
        print(f"\nReasoning accuracy: {reasoning_accuracy}")
    else:
        # Rankings
        ranking_accuracy, property_order_results = evaluate_ranking(data)
        print("\n")
        for k, v in ranking_accuracy.items():
            print(f"{k}: {v}")
        print(f"No rank sample output: {property_order_results['no_ranking']}")
        print(f"Invalid sample count: {property_order_results['invalid_ranking_count']}")