from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq 
from pydantic import BaseModel, Field
import operator
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize LLM
model = ChatGroq(
    model="llama-3.3-70b-versatile",  # or llama3-70b-8192, mixtral-8x7b-32768
    temperature=0.7,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

# Structured schema
class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback for the essay")
    score: int = Field(description="Score out of 9", ge=0, le=9)

structured_model = model.with_structured_output(EvaluationSchema)

# Define state
class IELTSState(TypedDict):
    essay: str
    language_feedback: str
    coherence_feedback: str
    lexical_feedback: str
    grammar_feedback: str
    overall_feedback: str
    individual_scores: Annotated[list[int], operator.add]
    avg_score: float

# Nodes
def evaluate_language(state: IELTSState):
    output = structured_model.invoke(
        f"Evaluate the language and vocabulary of this IELTS essay and score out of 9:\n\n{state['essay']}"
    )
    return {"language_feedback": output.feedback, "individual_scores": [output.score]}


def evaluate_coherence(state: IELTSState):
    output = structured_model.invoke(
        f"Evaluate the coherence and cohesion of this IELTS essay and score out of 9:\n\n{state['essay']}"
    )
    return {"coherence_feedback": output.feedback, "individual_scores": [output.score]}


def evaluate_lexical(state: IELTSState):
    output = structured_model.invoke(
        f"Evaluate the lexical resource of this IELTS essay and score out of 9:\n\n{state['essay']}"
    )
    return {"lexical_feedback": output.feedback, "individual_scores": [output.score]}


def evaluate_grammar(state: IELTSState):
    output = structured_model.invoke(
        f"Evaluate the grammatical range and accuracy of this IELTS essay and score out of 9:\n\n{state['essay']}"
    )
    return {"grammar_feedback": output.feedback, "individual_scores": [output.score]}
def round_to_ielts_band(score: float) -> float:
    """Round a float to the nearest valid IELTS band (increments of 0.5)."""
    return round(score * 2) / 2


def final_evaluation(state: IELTSState):
    prompt = f"""
    Summarize all IELTS feedbacks and generate an overall evaluation:
    - Language: {state['language_feedback']}
    - Coherence: {state['coherence_feedback']}
    - Lexical: {state['lexical_feedback']}
    - Grammar: {state['grammar_feedback']}
    """
    overall_feedback = model.invoke(prompt).content

    # Compute average and round properly
    avg_score_raw = sum(state["individual_scores"]) / len(state["individual_scores"])
    avg_score = round_to_ielts_band(avg_score_raw)

    return {"overall_feedback": overall_feedback, "avg_score": avg_score}


# Build graph
graph = StateGraph(IELTSState)
graph.add_node("language", evaluate_language)
graph.add_node("coherence", evaluate_coherence)
graph.add_node("lexical", evaluate_lexical)
graph.add_node("grammar", evaluate_grammar)
graph.add_node("final", final_evaluation)

graph.add_edge(START, "language")
graph.add_edge(START, "coherence")
graph.add_edge(START, "lexical")
graph.add_edge(START, "grammar")

graph.add_edge("language", "final")
graph.add_edge("coherence", "final")
graph.add_edge("lexical", "final")
graph.add_edge("grammar", "final")

graph.add_edge("final", END)

workflow = graph.compile()


def run_ielts_scorer(essay: str):
    return workflow.invoke({"essay": essay})
if __name__ == "__main__":
    sample_essay = """
    In recent years, the issue of climate change has become a significant concern worldwide. Many experts argue that human activities are the primary contributors to this phenomenon. This essay will discuss the causes of climate change and suggest possible solutions to mitigate its effects.

    One of the main causes of climate change is the excessive emission of greenhouse gases, such as carbon dioxide and methane, into the atmosphere. These gases trap heat from the sun, leading to a rise in global temperatures. Deforestation and industrial activities further exacerbate this problem by reducing the number of trees that can absorb carbon dioxide.

    To address climate change, governments and individuals must take proactive measures. Governments can implement policies that promote renewable energy sources, such as solar and wind power, to reduce reliance on fossil fuels. Additionally, individuals can contribute by adopting sustainable practices, such as reducing waste, conserving water, and using energy-efficient appliances.

    In conclusion, climate change is a pressing issue that requires immediate attention. By understanding its causes and implementing effective solutions, we can work towards a healthier planet for future generations.
    """
    result = run_ielts_scorer(sample_essay)
    print("Final Feedback:", result["overall_feedback"])
    print("Average Score:", result["avg_score"])
