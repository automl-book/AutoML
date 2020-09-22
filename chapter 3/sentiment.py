from google.cloud import automl

project_id = 'automlgcp'    # define your project id
model_id = 'TST8344486213285052416'   # define your model id
sentence = "An intermittently pleasing but mostly routine effort"

def predictSentiment():

  sentiment_prediction_model = automl.PredictionServiceClient()
  # Retrieves path of the model
  model_details = sentiment_prediction_model.model_path(project_id, 'us-central1', model_id)
  sentence_snippet = automl.types.TextSnippet( content=sentence,mime_type='text/plain')  
  payload_data = automl.types.ExamplePayload(text_snippet=sentence_snippet)
  predicted_response=sentiment_prediction_model.predict (model_details,    payload_data)
  for payload_result in predicted_response.payload:
    result= "Predicted sentiment score: {}"   .format(payload_result.text_sentiment.sentiment)
  return (result)

if __name__==’__main__’:
  predictSentiment()
