import numpy as np
import math
from PIL import Image
from extract import extraction 
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
from barcode import EAN13
from barcode.writer import ImageWriter

app = Flask(__name__)

# Get image features/ info
fe = extraction()
features=[]
img_paths=[]
barcodes=[]
scores=[]

for feature_path in Path("static/feature11").glob("*.npy"): # have to make features and img path for each file
    features.append(np.load(feature_path))
    img_paths.append(Path("static/set")/(feature_path.stem + ".jpg"))

features= np.array(features)# convert to numpy array

@app.route("/", methods=["GET", "POST"])


def index():
    if request.method == "POST":
        file = request.files["query_img"]
        # Save query img, read query image and save it on uploaded folder
        img= Image.open(file.stream) #PIL image
        uploaded_img_path= "static/uploaded/" +datetime.now().isoformat().replace(":",".") + "_" + file.filename
        img.save(uploaded_img_path)
        #make into numpy array and find sum of it and see what value we get
        query=np.array(fe.getInfo(img))
        # Find query value code
        sum=0
        x=0
        for i in query:
            x=x+1
            sum = sum + i 
        queryValue=sum/x #barcode value 
        queryValueBar=math.trunc(queryValue*1000000000000000)#make whole number
        # Check if query barcode equals any database barcode
        for img_path in sorted(Path("static/set").glob("*.jpg")):
             cursor= np.array(fe.getInfo(img=Image.open(img_path)))# Get img from set folder and get its info then turn into a numpy array
             #Find barcode value
             sum=0
             x=0
             for i in cursor:
                x=x+1
                sum = sum + i 
             barValue=sum/x
            # Find the mean value of the array
             barValueBar=math.trunc(barValue*1000000000000000)#make whole number
             barcodes.append(barValueBar)
             strDone= str(barValueBar)
             # Create image for barcode and save to root
             my_code = EAN13(strDone, writer=ImageWriter())
             my_code.save(strDone)
             # Check to see if query barcode equals any of given barcodes
             if queryValueBar==barValueBar:
                 found=img_path #get image path and break 
        #Determine which number was matched and reset values
        found=str(found)[11] 
        img_paths=[]
        feature_path=0
        features=[]
        for feature_path in Path("static/feature"+found).glob("*.npy"): # have to make features and img path for each file
            features.append(np.load(feature_path))
            img_paths.append(Path("static/"+found)/(feature_path.stem + ".jpg"))

        dists = np.linalg.norm(features - query, axis=1) # L2 distance to the features and normalize
        ids =np.argsort(dists)[:10] # get top 10 results
        results = [(dists[id], img_paths[id]) for id in ids] # Where we print all pictures
  
        return render_template("index2.html", query_path=uploaded_img_path, results=results) # send query and results/scores to display on html file 
    else: 
        return render_template("index2.html")

if __name__ == "__main__":
    app.run()
