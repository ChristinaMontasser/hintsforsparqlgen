from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate, 

)
from langchain_core.prompts import PromptTemplate

def build_prompt(prompt_type: str, system_message: str) -> ChatPromptTemplate:
    if prompt_type == "instructional":
        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                ("human",
                 """{user_message}

                Schema:
                {schema}

                Question:
                {question}

                SPARQL:
                """)
            ]
        )

    elif prompt_type == "structural":
        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                ("human", """{user_message}
                {question}

                """)
            ]
        )

    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
