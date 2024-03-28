import requests
from tqdm import tqdm
import json
from LLMedGenie.project_path import ROOT_DIR, OUTPUT_DIR
from LLMedGenie.data_validator import MedicalTranscriptValidator

backend_url = "http://127.0.0.1:5000"

labels = ['Surgery', 'Consult - History and Phy.', 'Cardiovascular / Pulmonary', 'Orthopedic', 'Radiology', 'General Medicine', 'Gastroenterology', 'Neurology', 'SOAP / Chart / Progress Notes', 'Obstetrics / Gynecology', 'Urology', 'Discharge Summary', 'ENT - Otolaryngology', 'Neurosurgery', 'Hematology - Oncology', 'Ophthalmology', 'Nephrology', 'Emergency Room Reports', 'Pediatrics - Neonatal',
          'Pain Management', 'Psychiatry / Psychology', 'Office Notes', 'Podiatry', 'Dermatology', 'Dentistry', 'Cosmetic / Plastic Surgery', 'Letters', 'Physical Medicine - Rehab', 'Sleep Medicine', 'Endocrinology', 'Bariatrics', 'IME-QME-Work Comp etc.', 'Chiropractic', 'Diets and Nutritions', 'Rheumatology', 'Speech - Language', 'Autopsy', 'Lab Medicine - Pathology', 'Allergy / Immunology', 'Hospice - Palliative Care']

validator = MedicalTranscriptValidator()

responses = []
inference_times = []
for label in tqdm(labels[:1], desc=f"Testing API"):
    request_body = {"prompt": label, "num_seq": 2, "max_tokens": 100}
    response = requests.post(
        f"{backend_url}/inference/batch/", json=request_body).json()
    sequences = response['Sequences']
    inference_time = response['Inference time']/request_body['num_seq']
    sequences = validator.validate(sequences)
    responses.extend(sequences)
    inference_times.append(inference_time)
inference_time_avg = sum(inference_times)/len(inference_times)

with open(f"{OUTPUT_DIR}/test.json", "w") as file:
    json.dump(responses, file, ensure_ascii=False, indent=2)

with open(f"{OUTPUT_DIR}/log.json", "w") as file:
    log = {"Valid JSON": validator.valid_json, "Invalid JSON": validator.invalid_json,
           "Valid Fields": validator.valid_field, "Invalid Fields": validator.invalid_field,
           "Average inference time": inference_time_avg}
    json.dump(log, file, ensure_ascii=False, indent=2)
