import json

def generate_clean_response(response):
    try:
        summary_response = response.strip('```').strip('json')
        cleaned_response = json.loads(summary_response)
        return cleaned_response
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}")
    except KeyError as e:
        raise ValueError(f"Missing required key in JSON: {str(e)}")