import requests
import json
import numpy as np
import random
import time
from tqdm import tqdm
import argparse
import random


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", default=2, type=int)
    parser.add_argument("--agents", default=2, type=int)
    parser.add_argument("--num_questions", default=1, type=int)
    parser.add_argument('--isModerator', action='store_true')
    parser.add_argument('--roleplay', action='store_true')
    parser.add_argument('--confidence', action='store_true')
    return parser.parse_args()

# pipe for my model


def generate_gsm(numagents, question):
    agent_contexts = [
        [{"model": f"model{i}", "content": f"Solve the following math problem: {question} Explain your reasoning and state your answer. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."}] for i in range(numagents)
        ]
    return agent_contexts

def generate_gsm_roleplay(numagents, question):
    agent_contexts = [
        [{"model": f"model{i}", "content": f"You are someone who failed grade school. Try to solve the following math problem: {question} Explain your reasoning and state your answer. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."}] for i in range(numagents)
        ]
    return agent_contexts

def generate_gsm_roleplay_confidence(numagents, question):
    agent_contexts = [
        [{"model": f"model{i}", "content": f"You are someone who failed grade school. Try to solve the following math problem: {question} Explain your reasoning and state your answer, along with a score from 1-5 (1 being unconfident) representing how confident you are in your answer. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."}] for i in range(numagents)
        ]
    return agent_contexts

def read_jsonl(path: str):
    with open(path, "r") as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def ask_agent(prompt, pipe):
    formatted_prompt = [{"role": "user", "content": prompt}]
    # generation_args = {
    #     "max_new_tokens": 500,
    #     "return_full_text": False,
    #     "temperature": 0.7, #creativity basically
    #     "do_sample": True, #greedy otherwise, give same res
    # }

    #make responses a bit more interesting
    generation_args_list = [{
        "max_new_tokens": 500,
        "return_full_text": False,
        "do_sample": True,
        "temperature": 0.6,            # Increase temperature for more randomness
        "top_p": 0.9,                 # Nucleus sampling with 95% probability mass
        "num_return_sequences": 1,     # Number of responses to generate
        "eos_token_id": pipe.tokenizer.eos_token_id,  # Ensure proper termination
    },
    {
        "max_new_tokens": 500,
        "return_full_text": False,
        "do_sample": True,
        "temperature": 0.7,            # Increase temperature for more randomness
        "top_p": 0.9,                 # Nucleus sampling with 95% probability mass
        "num_return_sequences": 1,     # Number of responses to generate
        "eos_token_id": pipe.tokenizer.eos_token_id,  # Ensure proper termination
    },
    {
        "max_new_tokens": 500,
        "return_full_text": False,
        "do_sample": True,
        "temperature": 0.8,            # Increase temperature for more randomness
        "top_p": 0.95,                 # Nucleus sampling with 95% probability mass
        "num_return_sequences": 1,     # Number of responses to generate
        "eos_token_id": pipe.tokenizer.eos_token_id,  # Ensure proper termination
    }
    ]
    generation_args = random.choice(generation_args_list)
     #for my reference: https://towardsdatascience.com/decoding-strategies-that-you-need-to-know-for-response-generation-ba95ee0faadc

    output = pipe(formatted_prompt, **generation_args)
    res = output[0]['generated_text']
    return res #text output

def ask_agent_last(prompt, pipe):
    formatted_prompt = [{"role": "user", "content": prompt}]
    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.7, #creativity basically
        "do_sample": True, #greedy otherwise, give same res
    }
    output = pipe(formatted_prompt, **generation_args)
    res = output[0]['generated_text']
    return res

def get_pipe():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct",
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    return pipe


if __name__ == "__main__":
    args = args_parse()

    num_agents = args.agents
    num_rounds = args.rounds
    random.seed(0)

    evaluation = args.num_questions

    #define pipe
    pipe = get_pipe()

    generated_description = []

    questions = read_jsonl("/content/gsm8k_test.jsonl")
    #random.shuffle(questions)

    for idx in tqdm(range(evaluation)):
        question = questions[idx]["question"]
        answer = questions[idx]["answer"]


        if args.roleplay and args.confidence:
          print("Taking role with confidence...")
          agent_contexts = generate_gsm_roleplay_confidence(num_agents, question)
        elif args.roleplay:
          print("Taking role...")
          agent_contexts = generate_gsm_roleplay(num_agents, question)
        else:
          agent_contexts = generate_gsm(num_agents, question) #generates the initial questions

        print(f"# Question No.{idx+1} starts...")

        message = []

        # Debate
        for round in range(num_rounds+1):
            # Refer to the summarized previous response
            if round != 0: #entering debate rounds
                feedback = "Based on the solutions from the other agents, update your own answer if you think you made a mistake in your calculations. If not, further reason why your original answer is correct."
                prev_responses = [agent[-1]["content"] for agent in agent_contexts]
                for i, agent_context in enumerate(agent_contexts):
                    other_responses = prev_responses[:i] + prev_responses[i+1:] #all other responses except own
                    if args.confidence:
                        prompt = f"Solve the following math problem: {question} Explain your reasoning and state your answer, along with a score from 1-5 (1 being unconfident) representing how confident you are in your answer. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."
                    else:
                        prompt = f"Solve the following math problem: {question} Explain your reasoning and state your answer. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response." #OG question
                    #add other agents response
                    if other_responses:
                        prompt += "\n\n Here are opinions from other agents and how confident they are in their answers:\n\n"
                        for i, other in enumerate(other_responses):
                            prompt += f"Agent{i}: {other}\n\n"

                    agent_context.append(
                        {"model": agent_context[-1]["model"], "content": prompt}
                    )


            for i, agent_context in enumerate(agent_contexts):
                agent_prompt = agent_context[-1]["content"] #the content
                text = ask_agent(agent_prompt, pipe) #get llm answer

                agent_context.append( #formatted so keep track of which model and which response
                    {"model": agent_context[-1]["model"], "content": text}
                )

        print(f"# Question No.{idx+1} debate is ended.")


        models_response = {
            f"model_{i}": [context[j]["content"] for j in range(1, len(context), 2)]
            for i, context in enumerate(agent_contexts)
        }
        allanswers = f"Here is the math problem: {question} "
        allanswers += "Here is a list of possible solutions given by other agents:\n "
        for i, model in enumerate(models_response):
            allanswers += f"Model {i+1}: {models_response[model][-1]}\n"
        allanswers += "Read the list of solutions above, and from them, determine which is correct. If none of them are, give your own final answer using the other response's logic as guidance. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response. \n"

        if args.isModerator:
            final_answer = ask_agent_last(allanswers, pipe)
            generated_description.append({"question_id": idx, "question": question, "all_agent_answers": models_response,"agent_response": allanswers, "final_answer": final_answer,"answer": answer})
        elif args.roleplay:
            if args.confidence: #has confidence
                p = f"You are someone who failed grade school. Try to solve the following math problem: {question} Explain your reasoning and state your answer, along with a score from 1-5 (1 being unconfident) representing how confident you are in your answer. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."
            else:
                p = f"You are someone who failed grade school. Try to solve the following math problem: {question} Explain your reasoning and state your answer. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."
            generated_description.append({"question_id": idx, "prompt": p, "question": question, "all_agent_answers": models_response,"agent_response": allanswers, "answer": answer})
        else:
            generated_description.append({"question_id": idx, "question": question, "all_agent_answers": models_response,"agent_response": allanswers, "answer": answer})


    with open("/content/gsm_res/gsm_{}_{}.json".format(num_agents, num_rounds), "w") as json_file:
        json.dump(generated_description, json_file, indent=4)
    print("All done!!!")