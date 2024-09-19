import json
import numpy as np
import time
import re
import argparse

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str)
    parser.add_argument("--num_agents", default=1, type=int)
    parser.add_argument("--savepath", default=None, type=str)
    parser.add_argument("--loadpath", default=None, type=str)
    parser.add_argument('--isModerator', action='store_true')
    parser.add_argument('--role', action='store_true')
    return parser.parse_args()

def parse_bullets(sentence):
    bullets_preprocess = sentence.split("\n")
    bullets = []

    for bullet in bullets_preprocess:
        try:
            idx = bullet.find(next(filter(str.isalpha, bullet)))
        except:
            continue

        bullet = bullet[idx:]

        if len(bullet) != 0:
            bullets.append(bullet)

    return bullets


def parse_yes_no(string):
    """
    Parses a string containing "yes" or "no" and returns a boolean value.

    Args:
        string (str): The string to parse.

    Returns:
        bool: True if the string contains "yes", False if the string contains "no".

    Raises:
        ValueError: If the input string does not contain "yes" or "no".
    """
    if "yes" in string.lower():
        return True
    elif "no" in string.lower():
        return False
    else:
        return None


def solve_math_problems(input_str):
    pattern = r"\d+\.?\d*"

    matches = re.findall(pattern, input_str)
    if matches:
        return matches[-1]

    return None

def parse_answer(input_str):
    pattern = r"\{([0-9.,$]*)\}"
    matches = re.findall(pattern, input_str)

    solution = None

    for match_str in matches[::-1]:
        solution = re.sub(r"[^0-9.]", "", match_str)
        if solution:
            break

    return solution


def compute_accuracy(gt, pred_solution):
    answers = solve_math_problems(gt)

    if answers is None:
        return None

    if type(pred_solution) == list:
        pred_answers = []

        for pred_solution in pred_solutions:
            pred_answer = parse_answer(pred_solution)

            if pred_answer is None:
                pred_answer = solve_math_problems(pred_solution)

            pred_answers.append(pred_answer)

        # print("pred_answers: ", pred_answers)

        pred_answer = most_frequent(pred_answers)
        # print("pred answer: ", pred_answer)
        # pred_answer = pred_answers[0]
    else:
        pred_answer = parse_answer(pred_solution)
        if pred_answer is None:
            pred_answer = solve_math_problems(pred_solution)

    if pred_answer is None:
        return float(pred_answer), False

    return float(pred_answer), float(answers) == float(pred_answer)
    # print(float(pred_answer), float(answers))
    # # try:
    # if float(answers) == float(pred_answer):
    #     return 1
    # else:
    #     return 0
    # except:
    #     import pdb
    #     pdb.set_trace()
    #     print(pred_solution)


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num

if __name__ == "__main__":
    args = args_parse()
    save_path = args.savepath
    load_path = args.loadpath
    isModerator = args.isModerator

    all_my_answers = []

    with open(f"{load_path}{args.filename}", "r") as f:
        response_dict = json.load(f)

    questions = [response_dict[i]["question"] for i in range(len(response_dict))] #list(response_dict.keys()) #all questions

    accuracies = []


    for i, question in enumerate(questions):

        responses = []
        # for response in response_dict:
        #       question_string = response["agent_response"][f"model_{round}"][-1]
        #       responses.append(question_string)
        response = response_dict[i]

        for agentnumber in range(args.num_agents):
            if args.role:
                for j in range(len(response["all_agent_answers"][f"model_{agentnumber}"])): #saved as wrong thing
                    question_string = response["all_agent_answers"][f"model_{agentnumber}"][j]
                    responses.append(question_string)


                pred_solutions = []
                for res in responses:
                    # print(response)
                    # pred_solution = response[-1]['content']
                    pred_solutions.append(res)
            elif not isModerator:
                for k in range(len(response["agent_response"][f"model_{agentnumber}"])):
                    question_string = response["agent_response"][f"model_{agentnumber}"][k]
                    responses.append(question_string)
                #print(f"here{agentnumber} {args.num_agents}", len(response["agent_response"][f"model_{agentnumber}"]))

                pred_solutions = []
                for res in responses:
                    # print(response)
                    # pred_solution = response[-1]['content']
                    pred_solutions.append(res)

        gt = response_dict[i]["answer"] #real answer
        #responses, gt = response_dict[question]
        if not isModerator:

            agent_answer, correctness = compute_accuracy(gt, pred_solutions)

        else:
            agent_answer, correctness = compute_accuracy(gt, response["final_answer"]) #for moderator experiement only

        print(f"Q{i} : " , agent_answer, correctness)
        if correctness:
            accuracies.append(1)
        else:
            accuracies.append(0)

        if not isModerator:
            res = {
            "question": question,
            "real_answer": gt,
            "agent_answer": agent_answer,
            "correct": correctness #true or false
        }
        else:
            res = {
            "question": question,
            "real_answer": gt,
            "agent_answer": agent_answer,
            "agent_responses_final": response["final_answer"], #has final response
            "correct": correctness #true or false
        }

        all_my_answers.append(res)


    print("accuracies:", np.mean(accuracies), np.std(accuracies) / (len(accuracies) ** 0.5))
    all_my_answers.append({"Total performance: " : np.mean(accuracies)})
    with open(f"{save_path}{args.filename}", "w") as f:
        json.dump(all_my_answers, f, indent=4)
    print("All done!!! + saved")
