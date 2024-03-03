from pydantic import BaseModel, field_validator, ValidationError
from typing import List
import json

class MedicalTranscript(BaseModel):
    medical_specialty: str
    description: str
    transcription: str
    sample_name: str
    keywords: List[str]    

    @field_validator("medical_specialty","description","transcription","sample_name",check_fields=True)
    @classmethod
    def validate_fields(cls, value):
        if not isinstance(value, str):
            raise ValueError('All fields must be strings')
        return value

    @field_validator("keywords",check_fields=True)
    @classmethod
    def validate_keywords(cls, value):
        if not all(isinstance(item, str) for item in value):
            raise ValueError('All items in the list must be strings')
        return value

    class Config:
        extra = "forbid"

class JsonValidator:
    def __init__(self):
        self.valid_json = 0
        self.invalid_json = 0
        self.valid_field = 0
        self.invalid_field = 0
    
    def validate_as_json(self, input_sequences):
        validated_json = []
        for input in input_sequences:
            try:
                input = json.loads(input['generated_text'].split("<|assistant|>")[1])
                self.valid_json += 1
                validated_json.append(input)                
            except json.JSONDecodeError as e:
                self.invalid_json += 1
        return validated_json
    
    def validate(self):
        raise NotImplementedError

class MedicalTranscriptValidator(JsonValidator):
    def __init__(self):
        super().__init__()

    def validate(self, input_sequences): 
        validated_json = self.validate_as_json(input_sequences)
        transcripts = []
        for transcript in validated_json:
            try:
                MedicalTranscript(**transcript)  
                self.valid_field += 1
                transcripts.append(transcript)              
            except ValidationError as e:
                self.invalid_field += 1
        return transcripts