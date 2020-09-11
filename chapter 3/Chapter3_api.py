from google.cloud import automl
from flask import Flask, request,jsonify

app=Flask(__name__)

@app.route ("/classification", methods=['POST'])
def Classification():
    try:
        json_data = request.get_json(force=True)
        project_id = json_data["project_id"]
        location = json_data["location"]
        model_id = json_data["model_id"]
        content = json_data["content"]
        result = []
        prediction_client = automl.PredictionServiceClient()
        model_full_id = prediction_client.model_path(
            project_id, location, model_id)
        text_snippet = automl.types.TextSnippet(
            content=content,mime_type='text/plain')
        payload = automl.types.ExamplePayload(text_snippet=text_snippet)
        response = prediction_client.predict(model_full_id, payload)
        for annotation_payload in response.payload:
            classification = {}
            classification["Class_Name"] = annotation_payload.display_name
            classification["Class_Score"] = annotation_payload.classification.score
            result.append(classification)
        result = {"results" : result}
        return jsonify(result)
    except Exception as e:
        return {"Error": str(e)}

if __name__ == "__main__" :
    app.run(port="5000")
