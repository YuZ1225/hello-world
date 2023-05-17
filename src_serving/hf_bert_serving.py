# -*- coding: utf-8 -*-
import allspark
import traceback
from PIL import Image
from io import BytesIO
import base64
import json

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification




class MyProcessor(allspark.BaseProcessor):
    """ MyProcessor is a example
        you can send mesage like this to predict
        curl -v http://127.0.0.1:8080/api/predict/service_name -d '2 105'
    """
    def initialize(self):
        """ load module, executed once at the start of the service
             do service intialization and load models in this function.
        """
        
        # the path of our model after mounted is `/eas/workspace/model/`
        model_name = '/eas/workspace/model/'
        
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.bert_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=model_name)

        
    def pre_process(self, data):
        """ data format pre process
        """
        return str(data, 'UTF-8')
        
    def post_process(self, data):
        """ process after process
        """

        return json.dumps(data, ensure_ascii=False)
    
    def process(self, data):
        """ process the request data
        """
        data = self.pre_process(data)
        result = self.bert_pipeline(data)
        
        if result:
            scores = result[0]['score']
            labels = result[0]['label']
            
            return self.post_process(result), 200
        else:
            return self.post_process("False"), 400
if __name__ == '__main__':
    # parameter worker_threads indicates concurrency of processing
    allspark.default_properties().put('rpc.keepalive', '100000')
    runner = MyProcessor(worker_threads=10, endpoint='0.0.0.0:8000')
    runner.run()
