#!/usr/bin/python
from flask import Flask
from flask_restplus import Api, Resource, fields
from sklearn.externals import joblib
from mov_model_deployment import predict_proba


app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='Genre Movie Prediction API',
    description='Genre Movie Prediction API')

ns = api.namespace('predict', 
     description='Movie Genre Classifier')
   
parser = api.parser()

parser.add_argument(
    'PLOT', 
    type=str, 
    required=True, 
    help='PLOT to be analyzed', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})

@ns.route('/')
class PredictApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        plots = args['PLOT']
        plots = plots.split('%3B')
        print(plots)
        
        return {
         "result": predict_proba(plots)
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5050)
