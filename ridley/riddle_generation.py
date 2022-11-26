import requests


def j1_jumbo(prompt, num_results):
    response = requests.post(
        "https://api.ai21.com/studio/v1/j1-jumbo/complete",
        headers={"Authorization": "6389NrbZgVb3k8AKFsKGX7sJjmcbzwXS"},  # API key
        json={
            "prompt": prompt,
            "numResults": num_results,
            "maxTokens": 200,
            "temperature": 0.7,
            "topKReturn": 0,
            "topP": 1,
            "countPenalty": {
                "scale": 0,
                "applyToNumbers": False,
                "applyToPunctuations": False,
                "applyToStopwords": False,
                "applyToWhitespaces": False,
                "applyToEmojis": False,
            },
            "frequencyPenalty": {
                "scale": 0,
                "applyToNumbers": False,
                "applyToPunctuations": False,
                "applyToStopwords": False,
                "applyToWhitespaces": False,
                "applyToEmojis": False,
            },
            "presencePenalty": {
                "scale": 0,
                "applyToNumbers": False,
                "applyToPunctuations": False,
                "applyToStopwords": False,
                "applyToWhitespaces": False,
                "applyToEmojis": False,
            },
            "stopSequences": [],
        },
    )
    return response
