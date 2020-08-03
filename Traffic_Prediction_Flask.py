# -*- coding: utf-8 -*-
"""

@author: amulc
"""

from flask import Flask, request
#TO generate UI for sending request via browser
from flasgger import Swagger 

from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2

app = Flask(__name__)

#Enable this app for swagger and it will auto generate UI
swagger = Swagger(app)

@app.route('/traffic', methods=['POST'])
def predict_traffic_file():
    #K.clear_session()

    #BELOW docstring lines are required to support swagger documentation
    """ Endpoint returning traffic sign image prediction
    ---
    parameters:
        - name: input_file
          in: formData
          type: file
          required: true
    """
    # Get the input file from the http request
    #read image file string data
    filestr = request.files['input_file'].read()
    #convert string data to numpy array
    npimg = np.fromstring(filestr, np.uint8)
    # convert numpy array to image
    img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
    in_image = cv2.resize(img, (32, 32), interpolation = cv2.INTER_AREA)
    
    in_image = np.expand_dims(in_image, axis = 0)

    # Load the saved traffic sign keras model
    model_filename = "model.h5"

    # Load model from file - read mode  
    traffic_model = load_model(model_filename, compile=False)

    # Make prediction using the input image file
    ## check dimensions before and after of the numpy array to see 2D, 3D, 4D
    result = traffic_model.predict(in_image)

    def sign(i):
        switcher={
                0:'Maximum speed limit (10 km/h)',
                1:'Maximum speed limit (30 km/h)',
                2:'Maximum speed limit (50 km/h)',
                3:'Maximum speed limit (60 km/h)',
                4:'Maximum speed limit (70 km/h)',
                5:'Maximum speed limit (80 km/h)',
                6:'End 80 km/h speed limit',
                7:'Maximum speed limit (100 km/h)',
                8:'Maximum speed limit (120 km/h)',
                9:'No Overtaking',
                10:'No overtaking by trucks/heavy goods vehicles',
                11:'Crossroads ahead with a minor road',
                12:'Priority road',
			    13:'Give way',
 			    14:'Stop',
			    15:'Road closed to all traffic',
			    16:'No trucks/heavy goods vehicles',
			    17:'No Entry',
			    18:'Attention: Other Dangers!',
			    19:'Curve to the left',
			    20:'Curve to the right',
			    21:'Series of curves, first to the left',
			    22:'Uneven road surface',
			    23:'Slippery road surface',
			    24:'Road narrows on the right',
			    25:'Roadworks',
			    26:'Traffic signals ahead',
			    27:'Pedestrian crossing ahead',
			    28:'Watch for children',
			    29:'Watch for cyclists',
			    30:'Risk of ice',
			    31:'Watch for wild animals',
			    32:'End all previously signed restrictions or prohibitions',
			    33:'Turn right only',
			    34:'Turn left only',
			    35:'Proceed straight ahead only',
			    36:'Proceed straight or turn right',
			    37:'Proceed straight or turn left',
			    38:'Keep right',
			    39:'Keep left',
			    40:'Roundabout',
			    41:'End overtaking prohibition',
			    42:'End overtaking prohibition for trucks/heavy goods vehicles',
		    }
        return switcher.get(i,"Unknown")
		
    print(sign(np.argmax(result)))

    K.clear_session()

    # Send the prediction as response
    return str(sign(np.argmax(result)))
	#return str(list(result))

if __name__ == '__main__':
    app.run(debug=True)
    