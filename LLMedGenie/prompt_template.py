import json
from langchain.prompts.prompt import PromptTemplate


def prompt_template_training(data: dict, template: str = "zephyr") -> list:
    prompt_template = None
    output = None
    if template == "llama2":
        prompt_template = PromptTemplate(
            input_variables=["system", "user", "assistant"], template="<s>[INST] <<SYS>>{system}<</SYS>>{user}[/INST]{assistant}</s>"
        )
    elif template == "zephyr":
        prompt_template = PromptTemplate(
            input_variables=["system", "user", "assistant"], template="<|system|>{system}</s><|user|>{user}</s><|assistant|>{assistant}"
        )
        output = [prompt_template.format(system=f"""You are a medical transcription generator. The User is going to ask you to generate a synthetic medical transcription of medical specialty type "{record['medical_specialty']}", and you will provide the answer in JSON format. The JSON should have 5 fields: "medical_specialty" that classifies the medical the medical specialty of the record, "description" that summarizes the transcription, "transcription" that is the main text of themedical record, "sample_name" that is the title of the transcription, "keywords" that provides a list of keywords relevant to the transcription""",
                                         user=record['medical_specialty'],
                                         assistant=json.dumps(record, indent=2))
                  for record in data]
    return output


def prompt_template_inference(input: str, template: str = "zephyr") -> str:
    prompt_template = None
    output = None
    if template == "llama2":
        prompt_template = PromptTemplate(
            input_variables=["system", "user"], template="<s>[INST] <<SYS>>{system}<</SYS>><<USER>>{user}<</USER>>[/INST]"
        )
    elif template == "zephyr":
        prompt_template = PromptTemplate(
            input_variables=["system", "user"], template="<|system|>{system}</s><|user|>{user}</s><|assistant|>"
        )
        output = prompt_template.format(system=f"""You are a medical transcription generator. The User is going to ask you to generate a synthetic medical transcription of medical specialty type "{input}", and you will provide the answer in JSON format. The JSON should have 5 fields: "medical_specialty" that classifies the medical the medical specialty of the record, "description" that summarizes the transcription, "transcription" that is the main text of the medical record, "sample_name" that is the title of the transcription, "keywords" that provides a list of keywords relevant to the transcription""",
                                        user=input)+f"""{{"medical_specialty":"{input}","""
    return output

# print(prompt_template_inferrence("Consult - History and Phy."))
