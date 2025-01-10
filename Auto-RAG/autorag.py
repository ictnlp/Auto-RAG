from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional, Generator

from omegaconf import MISSING

from flexrag.assistant import AssistantBase, ASSISTANTS
from flexrag.models import GenerationConfig, OpenAIGenerator, OpenAIGeneratorConfig
from flexrag.prompt import ChatPrompt, ChatTurn
from flexrag.retriever import DenseRetriever, DenseRetrieverConfig, RetrievedContext

from prompts import Knowledge_Prompt


@dataclass
class AutoRAGConfig(OpenAIGeneratorConfig, DenseRetrieverConfig):
    data_path: str = MISSING
    max_iter: int = 10
    elicit_max_iter: int = 5
    max_passages: int = 2
    verbose: bool = False


@ASSISTANTS("autorag", config_class=AutoRAGConfig)
class AutoRAGAssistant(AssistantBase):
    def __init__(self, cfg: AutoRAGConfig):
        self.prompt = ChatPrompt(
            system=(
                "Answer the question by retrieving external knowledge. "
                "Extract useful information from each retrieved document. "
                "If the information is insufficient or irrelevant, "
                "refine your query and search again until you are able to answer the question."
            )
        )
        # load model & retriever
        self.main_model = OpenAIGenerator(cfg)
        self.retriever = DenseRetriever(cfg)

        # set parameters
        self.verbose = cfg.verbose
        self.elicit_max_iter = cfg.elicit_max_iter
        self.max_iter = cfg.max_iter
        self.max_passages = cfg.max_passages
        return

    def interactive_answer(
        self, history: list[dict[str, Any]], show_details: bool
    ) -> Generator[tuple[list[dict], None], None, None]:
        # store the search history for interacting with the user
        question = history[-1]["content"].strip()
        if not show_details:
            history.append(
                {
                    "role": "assistant",
                    "content": "Retrieving and reasoning...",
                    "metadata": {"title": "🤖 Auto-RAG"},
                }
            )
            yield history, []

        queries = [question]
        retrieved_ids = []
        prompt = deepcopy(self.prompt)
        prompt.update(ChatTurn(role="user", content="Question: " + question.strip()))
        current_iter = 0
        first_model_output = None
        max_iter = self.max_iter
        # start retrieval iteration
        while max_iter > 0:
            if self.verbose:
                print("input", prompt)

            # generate thought & action
            first_model_output = self.main_model.chat(
                prompts=[prompt],
                generation_config=GenerationConfig(do_sample=False, max_new_tokens=200),
            )[0][0].strip()
            prompt.update(ChatTurn(role="assistant", content=first_model_output))
            history.append(
                {
                    "role": "assistant",
                    "content": first_model_output,
                    "metadata": {"title": "🤖 Auto-RAG"},
                }
            )
            if show_details:
                yield history, []

            # extract action
            if "Query:".lower() in first_model_output.lower():
                queries = [first_model_output.split("Query:")[-1].strip()]
                current_iter += 1
            elif "final answer" in first_model_output.lower():
                prompt.update(ChatTurn(role="assistant", content=first_model_output))
                break
            else:
                print("Exception: Follow Failed")
                print(prompt)
                print(first_model_output)

            # retrieve documents
            document = None
            queries[0] = queries[0].replace("[Dense]", "").strip()
            documents = []
            retrieval_results = self.retriever.search(queries[0])[0]

            # process retrieved documents
            for result in retrieval_results:
                if result.data["id"] not in retrieved_ids:
                    retrieved_ids.append(result.data["id"])
                    documents.append(result.data["text"].split("\n")[-1])
                if len(documents) >= self.max_passages:
                    break
            document = " ".join(documents)
            prompt.update(
                ChatTurn(
                    role="user",
                    content=f"Retrieved Document_{current_iter}: {document.strip()}",
                )
            )
            history.append(
                {
                    "role": "assistant",
                    "content": f"Retrieved Document_{current_iter}: {document.strip()}",
                    "metadata": {"title": "🔍︎ **Dense Retriever**"},
                }
            )
            if show_details:
                yield history, []

            max_iter -= 1

        first_model_output = ""
        if max_iter == 0:
            first_model_output = self.main_model.chat(
                prompts=[prompt],
                generation_config=GenerationConfig(temperature=0.0, max_new_tokens=150),
            )[0][0].strip()
            prompt.update(ChatTurn(role="assistant", content=first_model_output))
            history.append(
                {
                    "role": "assistant",
                    "content": first_model_output,
                    "metadata": {"title": "🤖 Auto-RAG"},
                }
            )
            if show_details:
                yield history, []

        max_iter = self.elicit_max_iter

        # try to generate pesudo document for answer the question
        while "Refined Query:" in first_model_output and max_iter > 0:
            current_iter += 1
            query = first_model_output.split("Refined Query:")[-1].strip()

            document_prompt = Knowledge_Prompt.format(question, query)

            document = self.main_model.generate(
                prefixes=[document_prompt],
                generation_config=GenerationConfig(
                    temperature=0.0,
                    max_new_tokens=200,
                    stop_str=["<|eot_id|>", "\n"],
                ),
            )[0][0].strip()

            # generate thought & action based on the pseudo document
            prompt.update(
                ChatTurn(
                    role="user",
                    content=f"Retrieved Document_{current_iter}: {document.strip()}",
                )
            )
            history.append(
                {
                    "role": "user",
                    "content": document.strip(),
                    "metadata": {"title": "Parametric Knowledge"},
                }
            )
            if show_details:
                yield history, []
            first_model_output = self.main_model.chat(
                prompts=[prompt],
                generation_config=GenerationConfig(
                    do_sample=False,
                    max_new_tokens=150,
                ),
            )[0][0].strip()
            prompt.update(ChatTurn(role="assistant", content=first_model_output))
            history.append(
                {
                    "role": "assistant",
                    "content": first_model_output,
                    "metadata": {"title": "🤖 Auto-RAG"},
                }
            )
            if show_details:
                yield history, []
            max_iter -= 1

        # Generate the final answer
        if not show_details:
            backup_history = []
            for id in range(len(history)):
                print(history[id])
                new_item = {}
                if type(history[id]) == dict:
                    new_item["role"] = history[id]["role"]
                    new_item["content"] = history[id]["content"]
                    if "metadata" in history[id]:
                        new_item["metadata"] = history[id]["metadata"]
                else:
                    new_item["role"] = history[id]["role"]
                    new_item["content"] = history[id]["content"]
                    if history[id]["metadata"]:
                        new_item["metadata"] = history[id]["metadata"]
                backup_history.append(new_item)
            history = [history[0], history[-1]]
            history[-1]["content"] = history[-1]["content"].split("Final Answer:")[-1].strip()
        else:
            backup_history = []
            backup_history.append(history[0])
            backup_history.append(
                {
                    "role": history[-1]["role"],
                    "content": history[-1]["content"].split("Final Answer:")[-1].strip(),
                    "metadata": history[-1]["metadata"],
                }
            )
        yield history, backup_history

    def answer(
        self, question: str
    ) -> tuple[str, Optional[list[RetrievedContext]], Optional[dict]]:
        queries = [question]
        retrieved_ids = []
        prompt = deepcopy(self.prompt)
        prompt.update(ChatTurn(role="user", content="Question: " + question.strip()))
        current_iter = 0
        first_model_output = None
        max_iter = self.max_iter
        response = ""
        # start retrieval iteration
        while max_iter > 0:
            if self.verbose:
                print("input", prompt)

            # generate thought & action
            first_model_output = self.main_model.chat(
                prompts=[prompt],
                generation_config=GenerationConfig(do_sample=False, max_new_tokens=200),
            )[0][0].strip()
            prompt.update(ChatTurn(role="assistant", content=first_model_output))

            # extract action
            if "Query:".lower() in first_model_output.lower():
                queries = [first_model_output.split("Query:")[-1].strip()]
                current_iter += 1
            elif "final answer" in first_model_output.lower():
                prompt.update(ChatTurn(role="assistant", content=first_model_output))
                response = first_model_output.split("Final Answer: ")[-1].strip()
                break
            else:
                print("Exception: Follow Failed")
                print(prompt)
                print(first_model_output)

            # retrieve documents
            document = None
            queries[0] = queries[0].replace("[Dense]", "").strip()
            documents = []
            retrieval_results = self.retriever.search(queries[0])[0]

            # process retrieved documents
            for result in retrieval_results:
                if result.data["id"] not in retrieved_ids:
                    retrieved_ids.append(result.data["id"])
                    documents.append(result.data["text"].split("\n")[-1])
                if len(documents) >= self.max_passages:
                    break
            document = " ".join(documents)
            prompt.update(
                ChatTurn(
                    role="user",
                    content=f"Retrieved Document_{current_iter}: {document.strip()}",
                )
            )
            max_iter -= 1

        first_model_output = ""
        if max_iter == 0:
            first_model_output = self.main_model.chat(
                prompts=[prompt],
                generation_config=GenerationConfig(temperature=0.0, max_new_tokens=150),
            )[0][0].strip()
            prompt.update(ChatTurn(role="assistant", content=first_model_output))

        # try to generate pesudo document for answer the question
        max_iter = self.elicit_max_iter
        while "Refined Query:" in first_model_output and max_iter > 0:
            current_iter += 1
            query = first_model_output.split("Refined Query:")[-1].strip()

            document_prompt = Knowledge_Prompt.format(question, query)

            document = self.main_model.generate(
                prefixes=[document_prompt],
                generation_config=GenerationConfig(
                    temperature=0.0,
                    max_new_tokens=200,
                    stop_str=["<|eot_id|>", "\n"],
                ),
            )[0][0].strip()

            # generate thought & action based on the pseudo document
            prompt.update(
                ChatTurn(
                    role="user",
                    content=f"Retrieved Document_{current_iter}: {document.strip()}",
                )
            )
            first_model_output = self.main_model.chat(
                prompts=[prompt],
                generation_config=GenerationConfig(
                    do_sample=False,
                    max_new_tokens=150,
                ),
            )[0][0].strip()
            prompt.update(ChatTurn(role="assistant", content=first_model_output))
            max_iter -= 1
        return (
            response,
            RetrievedContext(
                retriever="autorag", query=question, data={"text": document}
            ),
            {"prompt": prompt},
        )
